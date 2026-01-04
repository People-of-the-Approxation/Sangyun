# VerificationBERT.py
from typing import Optional, Tuple
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention

import UART_base
from Attention_approx import attention  # attention(Q,K,V,ser, return_attn=True) 필요


# =========================
# Custom Self-Attention (HW softmax)
# =========================
class BertSelfAttentionSoftmaxApprox(BertSelfAttention):
    """
    - FPGA(UART)로 softmax를 계산하는 attention 구현
    - output_attentions=True일 때 attention matrix를 self.last_attn에 저장
      self.last_attn shape: (B, H, T, T) numpy float64
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.ser = None
        self.last_attn: Optional[np.ndarray] = None  # (B,H,T,T)

    def set_serial(self, ser):
        self.ser = ser

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.ser is None:
            raise RuntimeError(
                "UART serial is not set. Call set_serial(ser) before forward()."
            )

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        def shape(x: torch.Tensor) -> torch.Tensor:
            return x.view(
                x.size(0),
                -1,
                self.num_attention_heads,
                self.attention_head_size,
            ).transpose(1, 2)

        query_layer = shape(mixed_query_layer)  # (B,H,T,Dh)
        key_layer = shape(mixed_key_layer)
        value_layer = shape(mixed_value_layer)

        B, H, T, Dh = query_layer.shape
        out = torch.zeros_like(query_layer)

        # attention_mask: 보통 (B,1,1,T) 형태, 값은 0 또는 큰 음수(-10000 등)
        mask_np = None
        if attention_mask is not None:
            mask = attention_mask.squeeze(1).squeeze(1)  # (B,T)
            mask_np = mask.detach().cpu().numpy()

        if output_attentions:
            self.last_attn = np.zeros((B, H, T, T), dtype=np.float64)
        else:
            self.last_attn = None

        for b in range(B):
            for h in range(H):
                Q_np = query_layer[b, h].detach().cpu().numpy()  # (T,Dh)
                K_np = key_layer[b, h].detach().cpu().numpy()  # (T,Dh)
                V_np = value_layer[b, h].detach().cpu().numpy()  # (T,Dh)

                if mask_np is not None:
                    # masked token의 V를 0으로 처리(너의 기존 스타일 유지)
                    Vm = V_np.copy()
                    Vm[mask_np[b] < 0] = 0.0
                    out_np, attn_np = attention(
                        Q_np, K_np, Vm, self.ser, return_attn=True
                    )
                else:
                    out_np, attn_np = attention(
                        Q_np, K_np, V_np, self.ser, return_attn=True
                    )

                out[b, h] = torch.tensor(
                    out_np, dtype=query_layer.dtype, device=query_layer.device
                )

                if output_attentions:
                    self.last_attn[b, h, :, :] = attn_np

        context_layer = out.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # transformers 규약상 attn tensor 반환할 수 있지만,
        # 여기서는 self.last_attn에 저장하고 반환은 None으로 유지.
        return context_layer, None


# =========================
# Model patch utilities
# =========================
def replace_self_attention(model: BertForSequenceClassification, NewSAClass):
    """
    BERT encoder의 self-attention 모듈을 NewSAClass로 교체
    """
    for layer in model.bert.encoder.layer:
        old_sa = layer.attention.self
        new_sa = NewSAClass(model.config)
        new_sa.load_state_dict(old_sa.state_dict(), strict=True)
        layer.attention.self = new_sa


def set_serial_to_model(model: BertForSequenceClassification, ser):
    """
    교체된 self-attention 모듈들에 UART serial 핸들 주입
    """
    for layer in model.bert.encoder.layer:
        sa = layer.attention.self
        if hasattr(sa, "set_serial"):
            sa.set_serial(ser)


def get_last_attention_matrix(
    model: BertForSequenceClassification, layer: int = 0, head: int = 0
) -> np.ndarray:
    """
    마지막 forward 호출에서 저장된 attention matrix를 가져옴.
    반환 shape: (T,T) (batch=1 기준)
    """
    L = len(model.bert.encoder.layer)
    layer = max(0, min(layer, L - 1))
    sa = model.bert.encoder.layer[layer].attention.self
    if not hasattr(sa, "last_attn") or sa.last_attn is None:
        raise RuntimeError(
            "No attention stored. Run forward with output_attentions=True first."
        )

    attn = sa.last_attn  # (B,H,T,T)
    B, H, T, _ = attn.shape
    if B < 1:
        raise RuntimeError("Invalid last_attn batch size.")

    head = max(0, min(head, H - 1))
    return np.asarray(attn[0, head], dtype=np.float64)


# =========================
# Factory (used by app.py)
# =========================
def build_models_sst2(device: str = "cpu"):
    """
    Returns:
      tokenizer
      baseline_model (SW attention)
      approx_model (HW attention: self-attention replaced)
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    baseline_model = (
        BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2"
        )
        .to(device)
        .eval()
    )

    approx_model = (
        BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2"
        )
        .to(device)
        .eval()
    )

    replace_self_attention(approx_model, BertSelfAttentionSoftmaxApprox)

    return tokenizer, baseline_model, approx_model


def compute_attention_hw(
    text: str,
    approx_model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    *,
    port: str,
    baud: int,
    layer: int = 0,
    head: int = 0,
    max_len: int = 128,
    device: str = "cpu",
):
    """
    FPGA(UART) 사용해서 attention matrix 계산
    Returns: (tokens, attn(T,T))
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    # UART open -> inject -> forward -> close
    ser = UART_base.open_serial(port, baud, timeout=1.0)
    try:
        set_serial_to_model(approx_model, ser)
        with torch.no_grad():
            _ = approx_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=(
                    inputs.get("attention_mask", None).to(device)
                    if inputs.get("attention_mask", None) is not None
                    else None
                ),
                output_attentions=True,
            )
        attn = get_last_attention_matrix(approx_model, layer=layer, head=head)
    finally:
        try:
            ser.close()
        except Exception:
            pass

    return tokens, attn


def compute_attention_sw(
    text: str,
    baseline_model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    *,
    layer: int = 0,
    head: int = 0,
    max_len: int = 128,
    device: str = "cpu",
):
    """
    SW(기본 BERT) attention matrix 계산
    Returns: (tokens, attn(T,T))
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=(
                inputs.get("attention_mask", None).to(device)
                if inputs.get("attention_mask", None) is not None
                else None
            ),
            output_attentions=True,
        )

    if out.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. (output_attentions=True) failed."
        )

    num_layers = len(out.attentions)
    layer = max(0, min(layer, num_layers - 1))
    attn_l = out.attentions[layer]  # (B, heads, T, T)
    heads = int(attn_l.shape[1])
    head = max(0, min(head, heads - 1))

    attn = attn_l[0, head].detach().cpu().numpy().astype(np.float64)
    return tokens, attn
