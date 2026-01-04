# VerificationGPT.py
from typing import Optional, Tuple
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import UART_base
from Attention_approx import attention  # attention(Q,K,V,ser, return_attn=True)

# transformers GPT2 attention class (version differences handled)
try:
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
except Exception as e:
    raise RuntimeError(
        "Cannot import GPT2Attention from transformers. Check your transformers version."
    ) from e


class GPT2AttentionSoftmaxApprox(GPT2Attention):
    """
    - FPGA(UART)로 softmax를 계산하는 GPT2 attention 구현
    - output_attentions=True일 때 attention matrix를 self.last_attn에 저장
      self.last_attn shape: (B, H, T, T) numpy float64

    유지하는 스타일:
    - padding mask는 'V를 0으로' 처리 (BERT 코드 스타일 유지)
    - causal(미래 토큰 참조 금지)은 score를 pad_value(-32.0)로 내려서 softmax에서 거의 0 되게 처리
      (HW softmax는 mask 연산을 모르므로, score를 낮춰서 마스킹 효과를 냄)
    """

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        # transformers 버전에 따라 init 시그니처가 달라서 안전하게 처리
        try:
            super().__init__(
                config, is_cross_attention=is_cross_attention, layer_idx=layer_idx
            )
        except TypeError:
            super().__init__(config, is_cross_attention=is_cross_attention)

        self.ser = None
        self.last_attn: Optional[np.ndarray] = None  # (B,H,T,T)

        # softmax 패딩값(=Q6.10에서 0x8000로 saturate 되는 -32.0)
        self.pad_value = -32.0

    def set_serial(self, ser):
        self.ser = ser

    @staticmethod
    def _shape_qkv(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        # x: (B,T,embed) -> (B,H,T,Dh)
        B, T, _ = x.shape
        return x.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        if self.ser is None:
            raise RuntimeError(
                "UART serial is not set. Call set_serial(ser) before forward()."
            )

        # ---- 원본 GPT2Attention 로직의 큰 흐름만 따라가서 Q,K,V 만든다 ----
        # c_attn: (B,T,3*embed) -> split
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)

        # shape to (B,H,T,Dh)
        query = self._shape_qkv(query, self.num_heads, self.head_dim)
        key = self._shape_qkv(key, self.num_heads, self.head_dim)
        value = self._shape_qkv(value, self.num_heads, self.head_dim)

        # past 처리(캐시) – 데모/검증 목적이면 보통 off지만, 안전하게 처리
        if layer_past is not None:
            past_key, past_value = layer_past
            # past_key/value: (B,H,Tpast,Dh)
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present = (key, value) if use_cache else None

        B, H, Tq, Dh = query.shape
        Tk = key.shape[2]

        # attention_mask: 보통 (B,1,1,Tk) 형태(또는 (B,1,Tk))
        mask_np = None
        if attention_mask is not None:
            # 가능한 형태들을 최대한 (B,Tk)로 만든다
            m = attention_mask
            while m.dim() > 2:
                m = m.squeeze(1)
            # 이제 (B,Tk) 기대. 값은 1/0 또는 0/큰음수 형태 모두 가능
            mask_np = m.detach().cpu().numpy()

        if output_attentions:
            self.last_attn = np.zeros((B, H, Tq, Tk), dtype=np.float64)
        else:
            self.last_attn = None

        # ---- HW softmax 기반 attention 계산 ----
        out = torch.zeros((B, H, Tq, Dh), device=query.device, dtype=query.dtype)

        # numpy 변환은 head별로 진행
        q_np_all = query.detach().cpu().numpy()
        k_np_all = key.detach().cpu().numpy()
        v_np_all = value.detach().cpu().numpy()

        for b in range(B):
            # padding mask 반영(기존 스타일 유지): masked token의 V를 0으로
            if mask_np is not None:
                # mask가 0/1이면 0인 곳을 masked로 취급.
                # mask가 0/큰음수이면 큰음수(<0)인 곳을 masked로 취급.
                if np.issubdtype(mask_np.dtype, np.floating) or np.issubdtype(
                    mask_np.dtype, np.integer
                ):
                    # heuristic
                    masked = (mask_np[b] == 0) | (mask_np[b] < 0)
                else:
                    masked = mask_np[b] < 0
            else:
                masked = None

            for h in range(H):
                Q = q_np_all[b, h]  # (Tq,Dh)
                K = k_np_all[b, h]  # (Tk,Dh)
                V = v_np_all[b, h]  # (Tk,Dh)

                if masked is not None:
                    Vm = V.copy()
                    Vm[masked] = 0.0
                else:
                    Vm = V

                # causal mask(미래 토큰 금지):
                # attention()은 mask 개념이 없으니, score를 pad_value(-32)로 깔아서 softmax에서 거의 0이 되게 만든다.
                # 이를 위해 attention() 내부에서 쓰는 scores = (K@Q[i])/sqrt(dk) 전에 직접 score를 만들기 어렵지만,
                # attention()을 그대로 쓰려면 'K'를 변형할 수밖에 없다.
                #
                # 여기서는 더 안전하게: attention()을 쓰지 않고, row마다 score를 만들어 softmax_fpga_variable에 태우는 방식이 필요하지만
                # 네 코드의 "attention()"은 이미 HW softmax 호출을 포함하고 있고 row-loop 구조라서,
                # causal mask 처리는 attention() 안의 softmax input vector(scores)에 적용하는 형태로 바꾸는 게 맞다.
                #
                # => 최소 변경으로 유지하기 위해: 아래처럼 attention()을 "row별로 score를 만든 뒤" softmax_fpga_variable을 부르는 구조로 두고,
                # causal mask는 score[i, j>i]를 pad_value로 치환한다.
                #
                # 그래서 여기서는 Attention_approx.attention() 대신, 동일한 로직을 이 클래스 안에서 row-by-row로 실행한다.

                Nq = Q.shape[0]
                Nk = K.shape[0]

                attn_mat = (
                    np.zeros((Nq, Nk), dtype=np.float64) if output_attentions else None
                )
                out_np = np.zeros((Nq, Vm.shape[1]), dtype=np.float64)

                # row-wise
                for i in range(Nq):
                    scores = (K @ Q[i].T) / np.sqrt(Dh)  # (Nk,)

                    # causal: j > (i + past_len) 를 막아야 함.
                    # 현재 key는 past 포함 길이 Tk, query는 현재 시점 길이 Tq.
                    # 간단히: query index i가 key index 기준으로 "끝쪽"에 해당한다고 보고,
                    #         허용되는 key는 [0 .. (Tk - Tq + i)]까지.
                    # past_len = Tk - Tq
                    past_len = Tk - Tq
                    allowed_max = past_len + i
                    if allowed_max + 1 < Nk:
                        scores[allowed_max + 1 :] = self.pad_value

                    # HW softmax 호출: Attention_approx.softmax_fpga_variable을 통해 들어감
                    # (pad_value=-32.0 유지)
                    from softmax_packet import softmax_fpga_variable

                    probs = softmax_fpga_variable(
                        self.ser,
                        scores,
                        pad_value=self.pad_value,
                        deadline_s=2.0,
                    )  # (Nk,)

                    if output_attentions:
                        attn_mat[i, :] = probs

                    out_np[i, :] = np.asarray(probs, dtype=np.float64) @ Vm

                out[b, h] = torch.tensor(out_np, device=query.device, dtype=query.dtype)

                if output_attentions:
                    self.last_attn[b, h, :, :] = attn_mat

        # merge heads: (B,H,T,Dh)->(B,T,embed)
        context = out.permute(0, 2, 1, 3).contiguous().view(B, Tq, H * Dh)
        # output projection
        attn_output = self.c_proj(context)
        attn_output = self.resid_dropout(attn_output)

        # transformers 관례: (attn_output, present, attn_weights) 형태
        # 우리는 HW attention을 self.last_attn에 저장하고, return_attentions는 None로 둬도 됨.
        # 하지만 GPT2 모델이 out.attentions를 구성하려면 attn_weights를 반환해야 함.
        # -> SW baseline은 원래 모델로 attentions 받으면 되고,
        # -> HW approx는 get_last_attention_matrix로 가져오니 여기서는 None 반환 유지 가능.
        attn_weights = None
        return attn_output, present, attn_weights


def replace_gpt2_attention(model: torch.nn.Module, NewAttnClass):
    """
    GPT2 transformer blocks의 attention을 NewAttnClass로 교체
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError(
            "This model does not look like a GPT-2 architecture (missing transformer.h)."
        )

    for idx, block in enumerate(model.transformer.h):
        old_attn = block.attn
        try:
            new_attn = NewAttnClass(
                model.config, is_cross_attention=False, layer_idx=idx
            )
        except TypeError:
            new_attn = NewAttnClass(model.config)

        new_attn.load_state_dict(old_attn.state_dict(), strict=True)
        block.attn = new_attn


def set_serial_to_model(model: torch.nn.Module, ser):
    """
    교체된 attention 모듈들에 UART serial 핸들 주입
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise RuntimeError("Model has no transformer.h blocks.")

    for block in model.transformer.h:
        attn = block.attn
        if hasattr(attn, "set_serial"):
            attn.set_serial(ser)


def get_last_attention_matrix(
    model: torch.nn.Module, layer: int = 0, head: int = 0
) -> np.ndarray:
    """
    마지막 forward 호출에서 저장된 attention matrix를 가져옴.
    반환 shape: (T,T) (batch=1 기준)
    """
    blocks = model.transformer.h
    L = len(blocks)
    layer = max(0, min(int(layer), L - 1))

    attn = blocks[layer].attn
    if not hasattr(attn, "last_attn") or attn.last_attn is None:
        raise RuntimeError(
            "No attention stored. Run forward with output_attentions=True first."
        )

    a = attn.last_attn  # (B,H,T,Tk)
    B, H, T, Tk = a.shape
    if B < 1:
        raise RuntimeError("Invalid last_attn batch size.")
    head = max(0, min(int(head), H - 1))
    return np.asarray(a[0, head], dtype=np.float64)


def build_models_gpt(device: str = "cpu", model_name: str = "gpt2"):
    """
    Returns:
      tokenizer
      baseline_model (SW attention)
      approx_model (HW attention: attention replaced)

    NOTE:
    - GPT 계열은 'SST-2로 파인튜닝된 GPT2 분류 모델'을 쓰는 게 정석.
    - model_name을 네가 쓰는 체크포인트로 바꿔도 됨.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2는 pad_token이 없어서 attention_mask/padding 쓸 때 필요
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_model = (
        AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    )

    approx_model = (
        AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    )

    replace_gpt2_attention(approx_model, GPT2AttentionSoftmaxApprox)
    return tokenizer, baseline_model, approx_model
