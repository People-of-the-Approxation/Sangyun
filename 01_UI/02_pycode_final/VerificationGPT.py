# VerificationGPT.py
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from softmax_packet import softmax_fpga_variable

try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
except Exception as e:
    raise RuntimeError(
        "Cannot import GPT2Attention. Check your transformers version."
    ) from e


class GPT2AttentionSoftmaxApprox(GPT2Attention):
    """
    Optimized GPT-2 Attention (Hybrid):
    - Default: Pure PyTorch attention (fast)
    - HW Mode: Apply FPGA(UART) softmax ONLY on ONE selected layer
      and ONLY on one row (row = Tq - 1, i.e., current token row).
      By default, HW layer is `store_layer` (same as heatmap target layer).
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        try:
            super().__init__(
                config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
            )
        except TypeError:
            super().__init__(config, is_cross_attention=is_cross_attention)

        self.ser = None
        self.last_attn: Optional[np.ndarray] = None
        self.force_store_attn: bool = False
        self.pad_value = -32.0

        # 저장 제어 플래그 (UI에서 layer/head 선택에 이미 사용 중)
        self.store_only: bool = False
        self.store_layer: int = 0
        self.store_head: int = 0

        # HW 적용 제어 (기본은 store_layer를 HW layer로 재사용)
        # 필요하면 나중에 별도 setter로 분리 가능
        self.hw_only_one_layer: bool = True  # ✅ "레이어 1개만 HW" 옵션
        self.hw_layer: int = (
            0  # ✅ HW 적용 레이어 (기본값; set_store_target에서 동기화)
        )

    def set_serial(self, ser):
        self.ser = ser

    def set_force_store_attn(self, flag: bool):
        self.force_store_attn = bool(flag)

    def set_store_target(self, layer: int, head: int, store_only: bool = True):
        self.store_only = bool(store_only)
        self.store_layer = int(layer)
        self.store_head = int(head)

        # ✅ heatmap target layer를 HW layer로도 사용 (UI 연동 간단)
        self.hw_layer = int(layer)

    @staticmethod
    def _shape_qkv(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # (B, T, Embed) -> (B, H, T, Dh)
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # 1. Attention 저장 여부 판단
        want_attn = bool(output_attentions) or bool(
            kwargs.get("output_attentions", False)
        )
        want_attn = want_attn or getattr(self, "force_store_attn", False)

        # 2. Q, K, V 추출 (PyTorch Tensor 유지)
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        query = self._shape_qkv(query, self.num_heads, self.head_dim)
        key = self._shape_qkv(key, self.num_heads, self.head_dim)
        value = self._shape_qkv(value, self.num_heads, self.head_dim)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present = (key, value) if use_cache else None

        query_layer = query
        key_layer = key
        value_layer = value

        B, H, Tq, Dh = query_layer.shape
        Tk = key_layer.shape[2]

        # 3. Score 계산
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_weights = attn_weights / (float(Dh) ** 0.5)

        # 4. Causal Mask (Prompt에서는 필요, Generation(Tq=1)에서는 불필요)
        if Tq > 1:
            causal_mask = torch.triu(
                torch.ones((Tq, Tk), dtype=torch.bool, device=attn_weights.device),
                diagonal=Tk - Tq + 1,
            )
            attn_weights.masked_fill_(causal_mask[None, None, :, :], self.pad_value)

        # 5. Attention Mask (Padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                _mask = attention_mask[:, None, None, :]
            else:
                _mask = attention_mask

            attn_weights = torch.where(
                _mask > 0,
                attn_weights,
                torch.tensor(
                    self.pad_value, dtype=attn_weights.dtype, device=attn_weights.device
                ),
            )

        # 6. Softmax (SW, fast)
        attn_probs = F.softmax(attn_weights, dim=-1)

        # ==========================================================
        # 🚀 [HW Hybrid Logic] "선택된 1개 레이어"에서만 HW softmax 적용
        #     - row_idx = Tq - 1 (현재 토큰 row)
        #     - head는 전체 유지 (요청대로)
        # ==========================================================
        if self.ser is not None:
            this_layer_idx = getattr(self, "layer_idx", None)

            apply_hw = True
            if self.hw_only_one_layer:
                apply_hw = (this_layer_idx is not None) and (
                    int(this_layer_idx) == int(self.hw_layer)
                )

            if apply_hw and Tq >= 1:
                b_idx = 0
                row_idx = Tq - 1  # ✅ 현재 토큰 row (prompt: 마지막 토큰, gen: 0)

                # (H, Tk) 가져와서 한 번만 CPU로 이동
                row_scores_tensor = attn_weights[b_idx, :, row_idx, :]  # (H, Tk)
                row_scores_np = row_scores_tensor.detach().cpu().numpy()

                hw_probs_np = np.zeros_like(row_scores_np)

                for h in range(H):
                    try:
                        hw_out = softmax_fpga_variable(
                            self.ser,
                            row_scores_np[h],
                            pad_value=self.pad_value,
                            deadline_s=2.0,
                        )
                        hw_probs_np[h] = hw_out
                    except Exception:
                        # fallback: SW softmax 결과
                        fallback = (
                            attn_probs[b_idx, h, row_idx, :].detach().cpu().numpy()
                        )
                        hw_probs_np[h] = fallback

                hw_probs_tensor = (
                    torch.from_numpy(hw_probs_np)
                    .to(attn_probs.device)
                    .type(attn_probs.dtype)
                )
                attn_probs[b_idx, :, row_idx, :] = hw_probs_tensor

        # 7. Dropout & Weighted Sum
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_layer)  # (B, H, Tq, Dh)

        # 8. Heatmap 저장 (Target Layer/Head만)
        this_layer_idx = getattr(self, "layer_idx", None)
        store_this = (
            want_attn
            and self.store_only
            and (this_layer_idx is not None)
            and (int(this_layer_idx) == int(self.store_layer))
        )

        if store_this:
            target_head = self.store_head
            saved_map = attn_probs[0, target_head, :, :].detach().cpu().numpy()
            self.last_attn = saved_map.astype(np.float64)

        # 9. Output Format (B, Tq, H*Dh)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        new_shape = attn_output.size()[:-2] + (self.num_heads * self.head_dim,)
        attn_output = attn_output.view(*new_shape)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None


def replace_gpt2_attention(model: torch.nn.Module, NewAttnClass):
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("Model is not GPT-2 style.")
    for idx, block in enumerate(model.transformer.h):
        old_attn = block.attn
        new_attn = NewAttnClass(model.config, is_cross_attention=False, layer_idx=idx)
        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        block.attn = new_attn


def set_serial_to_model(model: torch.nn.Module, ser):
    for block in model.transformer.h:
        if hasattr(block.attn, "set_serial"):
            block.attn.set_serial(ser)


def clear_serial_from_model(model: torch.nn.Module):
    for block in model.transformer.h:
        if hasattr(block.attn, "set_serial"):
            block.attn.set_serial(None)


def get_last_attention_matrix(model, layer=0, head=0):
    layer = max(0, min(int(layer), len(model.transformer.h) - 1))
    attn_mod = model.transformer.h[layer].attn

    if hasattr(attn_mod, "last_attn") and attn_mod.last_attn is not None:
        return attn_mod.last_attn

    return np.zeros((1, 1), dtype=np.float64)


def set_force_store_attn_to_model(model: torch.nn.Module, flag: bool):
    for block in model.transformer.h:
        if hasattr(block.attn, "set_force_store_attn"):
            block.attn.set_force_store_attn(flag)


def set_store_target_to_model(
    model: torch.nn.Module, layer: int, head: int, store_only: bool = True
):
    for block in model.transformer.h:
        if hasattr(block.attn, "set_store_target"):
            block.attn.set_store_target(layer, head, store_only)
