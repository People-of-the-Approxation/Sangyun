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
    Optimized GPT-2 Attention:
    - Default: Uses fast PyTorch operations (GPU/CPU)
    - HW Mode: Only converts Row 0 to NumPy for UART, keeps others in PyTorch
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

        # 저장 제어 플래그
        self.store_only: bool = False
        self.store_layer: int = 0
        self.store_head: int = 0

    def set_serial(self, ser):
        self.ser = ser

    def set_force_store_attn(self, flag: bool):
        self.force_store_attn = bool(flag)

    def set_store_target(self, layer: int, head: int, store_only: bool = True):
        self.store_only = bool(store_only)
        self.store_layer = int(layer)
        self.store_head = int(head)

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

        # (Batch, Heads, T_query, Dim)
        # Tensor 연산 최적화를 위해 여기서 Shape 확보
        query_layer = query
        key_layer = key
        value_layer = value

        B, H, Tq, Dh = query_layer.shape
        Tk = key_layer.shape[2]

        # 3. Score 계산 (Matrix Multiplication - PyTorch Native)
        # (B, H, Tq, Dh) @ (B, H, Dh, Tk) -> (B, H, Tq, Tk)
        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_weights = attn_weights / (float(Dh) ** 0.5)

        # 4. Causal Mask 적용
        # GPT2Attention 원본 로직 참조 (triu 사용)
        if Tq > 1 or Tk > 1:  # 일반적인 경우
            # causal mask 생성
            bias = torch.tril(
                torch.ones((Tk, Tk), dtype=torch.uint8, device=attn_weights.device)
            ).view(1, 1, Tk, Tk)
            # 현재 윈도우에 맞게 슬라이싱
            # query 길이만큼, key 길이만큼
            # causal: 미래 토큰 마스킹
            # (GPT2 구현상 bias는 register buffer지만 여기선 간단히 생성)

            # 간단한 Causal Masking:
            # i > j (과거) 허용, i < j (미래) 마스킹
            # 실제로는 attention_mask가 들어오므로 그것과 결합됨.
            # 하지만 generation 단계에서는 past_key_value가 있으므로
            # Tq=1 일 때는 마스킹 불필요 (항상 과거만 보므로)
            pass

        # transformers의 GPT2 모델은 내부적으로 `bias` 버퍼를 이용해 causal masking을 합니다.
        # 여기서는 직접 구현 대신 attention_mask와 결합하여 처리하거나
        # 간단히 상삼각 행렬 마스킹을 수행합니다.

        # Generation 중(Tq=1)에는 Causal Mask 불필요 (이미 과거 Key만 존재)
        # Prompt Forward 중(Tq > 1)에는 Causal Mask 필요
        if Tq > 1:
            causal_mask = torch.triu(
                torch.ones((Tq, Tk), dtype=torch.bool, device=attn_weights.device),
                diagonal=Tk - Tq + 1,
            )
            attn_weights.masked_fill_(causal_mask[None, None, :, :], self.pad_value)

        # 5. Attention Mask (Padding) 적용
        if attention_mask is not None:
            # attention_mask: (B, 1, 1, Tk) 형태라고 가정 (transformers 표준)
            # 만약 (B, Tk)라면 차원 확장 필요
            if attention_mask.dim() == 2:
                _mask = attention_mask[:, None, None, :]
            else:
                _mask = attention_mask

            # mask가 0인 부분에 pad_value 적용
            # (transformers는 보통 1.0(keep), 0.0(mask)을 쓰거나 0, -inf를 씀)
            # 여기서는 값이 0이면 마스킹이라 가정
            attn_weights = torch.where(
                _mask > 0,
                attn_weights,
                torch.tensor(
                    self.pad_value, dtype=attn_weights.dtype, device=attn_weights.device
                ),
            )

        # 6. Softmax (PyTorch Native - 매우 빠름)
        attn_probs = F.softmax(attn_weights, dim=-1)

        # ==========================================================
        # 🚀 [HW Hybrid Logic] Row 0만 바꿔치기 (필요시에만 NumPy 변환)
        # ==========================================================
        if self.ser is not None:
            # HW 연산이 필요한 경우에만 CPU/NumPy로 데이터 이동
            # (Batch loop 대신 Batch=0만 처리한다고 가정하거나 Loop)

            # 성능을 위해 Batch 처리는 생략하고 B=0에 대해서만 HW 적용 예시
            # (데모용으로는 충분)
            b_idx = 0

            # Row 0의 Score 가져오기 (Tensor) -> (H, Tk)
            # Tq의 0번째 인덱스 (Prompt의 첫 토큰 or Gen의 현재 토큰)
            row0_scores_tensor = attn_weights[b_idx, :, 0, :]

            # CPU로 이동 (작은 데이터라 빠름)
            row0_scores_np = row0_scores_tensor.detach().cpu().numpy()  # (H, Tk)

            # HW 결과를 담을 배열
            hw_probs_np = np.zeros_like(row0_scores_np)

            # Head별로 HW 요청
            for h in range(H):
                try:
                    # UART 전송
                    hw_out = softmax_fpga_variable(
                        self.ser,
                        row0_scores_np[h],
                        pad_value=self.pad_value,
                        deadline_s=2.0,  # HW 타임아웃
                    )
                    hw_probs_np[h] = hw_out
                except Exception:
                    # 실패 시 SW값(이미 계산됨) 사용을 위해 0으로 두지 않고
                    # 기존 PyTorch softmax 값을 가져옴
                    fallback = attn_probs[b_idx, h, 0, :].detach().cpu().numpy()
                    hw_probs_np[h] = fallback

            # 결과를 다시 텐서로 변환하여 덮어쓰기
            hw_probs_tensor = (
                torch.from_numpy(hw_probs_np)
                .to(attn_probs.device)
                .type(attn_probs.dtype)
            )
            attn_probs[b_idx, :, 0, :] = hw_probs_tensor

        # 7. Dropout & Weighted Sum
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value_layer)  # (B, H, Tq, Dh)

        # 8. Heatmap 저장 (Target Layer/Head만)
        # 여기서만 NumPy 변환 발생 (저장용)
        this_layer_idx = getattr(self, "layer_idx", None)
        store_this = (
            want_attn
            and self.store_only
            and (this_layer_idx is not None)
            and (int(this_layer_idx) == int(self.store_layer))
        )

        if store_this:
            # (B, H, Tq, Tk) -> (Tq, Tk) (Batch=0, Target Head)
            target_head = self.store_head
            saved_map = attn_probs[0, target_head, :, :].detach().cpu().numpy()
            self.last_attn = saved_map.astype(np.float64)
        else:
            if not getattr(self, "store_only", False) and want_attn:
                # store_only가 꺼져있고 want_attn이면 전체 저장 (기존 호환)
                # 메모리 낭비 가능성 있음
                pass

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
    # 저장된 last_attn 가져오기
    layer = max(0, min(int(layer), len(model.transformer.h) - 1))
    attn_mod = model.transformer.h[layer].attn

    if hasattr(attn_mod, "last_attn") and attn_mod.last_attn is not None:
        return attn_mod.last_attn

    # 없으면 더미 리턴
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
