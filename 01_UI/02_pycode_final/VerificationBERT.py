from typing import Optional, Tuple
import torch
import numpy as np
import datasets
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from Attention_approx import attention
import UART_base


# =========================
# Custom Self-Attention (HW softmax)
# =========================
class BertSelfAttentionSoftmaxApprox(BertSelfAttention):
    """
    FPGA(UART)로 softmax를 계산하는 attention 구현.
    웹 UI의 compute_hw_all에서 호출됩니다.
    """

    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.ser = None
        self.last_attn: Optional[np.ndarray] = None

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
        **kwargs,  # [중요] transformers 버전 호환성 (past_key_values 등 수용)
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

        query_layer = shape(mixed_query_layer)
        key_layer = shape(mixed_key_layer)
        value_layer = shape(mixed_value_layer)

        B, H, T, Dh = query_layer.shape
        out = torch.zeros_like(query_layer)

        # 마스크 처리 (필요시 사용)
        mask_np = None
        if attention_mask is not None:
            mask = attention_mask.squeeze(1).squeeze(1)
            mask_np = mask.detach().cpu().numpy()

        if output_attentions:
            self.last_attn = np.zeros((B, H, T, T), dtype=np.float64)
        else:
            self.last_attn = None

        for b in range(B):
            for h in range(H):
                Q_np = query_layer[b, h].detach().cpu().numpy()
                K_np = key_layer[b, h].detach().cpu().numpy()
                V_np = value_layer[b, h].detach().cpu().numpy()

                # 마스크가 있을 경우 처리 (간단하게 0 처리 예시)
                # 실제로는 -inf 등으로 처리되지만 여기선 HW 특성에 맞김
                # (Attention_approx.py가 이를 받아서 처리)

                # FPGA Attention 호출 (return_attn=True로 매트릭스 받아옴)
                out_np, attn_np = attention(
                    Q_np, K_np, V_np, self.ser, return_attn=True
                )

                out[b, h] = torch.tensor(
                    out_np, dtype=query_layer.dtype, device=query_layer.device
                )

                if output_attentions:
                    self.last_attn[b, h, :, :] = attn_np

        context_layer = out.transpose(1, 2).contiguous().view(B, T, H * Dh)

        # transformers 규약에 맞춰 반환
        return context_layer, None


# =========================
# Model Utility Functions
# =========================


def replace_self_attention(model: BertForSequenceClassification, NewSAClass):
    for layer in model.bert.encoder.layer:
        old_sa = layer.attention.self
        new_sa = NewSAClass(model.config)
        new_sa.load_state_dict(old_sa.state_dict(), strict=True)
        layer.attention.self = new_sa


def set_serial_to_model(model: BertForSequenceClassification, ser):
    for layer in model.bert.encoder.layer:
        sa = layer.attention.self
        if hasattr(sa, "set_serial"):
            sa.set_serial(ser)


def get_last_attention_matrix(model, layer=0, head=0):
    L = len(model.bert.encoder.layer)
    layer = max(0, min(layer, L - 1))
    sa = model.bert.encoder.layer[layer].attention.self

    if not hasattr(sa, "last_attn") or sa.last_attn is None:
        return None

    attn = sa.last_attn  # (B, H, T, T)
    H = attn.shape[1]
    head = max(0, min(head, H - 1))

    return attn[0, head]  # Batch 0번의 특정 Head 반환


def build_models_sst2(device="cpu"):
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
