from typing import Optional, Tuple
import serial
import torch
import numpy as np
import datasets
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from attention_approx import attention
from softmax_batch import open_serial, close_serial


class BertSelfAttentionSoftmaxApprox(BertSelfAttention):

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
        **kwargs,
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

                out_np = attention(
                    Q_np, K_np, V_np, self.ser, pad_value=-32.0, timeout_s=2.0
                )
                out[b, h] = torch.tensor(
                    out_np, dtype=query_layer.dtype, device=query_layer.device
                )

        context_layer = out.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return context_layer, None


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

    attn = sa.last_attn
    H = attn.shape[1]
    head = max(0, min(head, H - 1))

    return attn[0, head]


def build_model_BERT(ser: serial.Serial):
    device = "cpu"

    print(f"Loading BERT model for SST-2...")
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
    set_serial_to_model(approx_model, ser)

    return tokenizer, baseline_model, approx_model, device


def evaluate_SST2():
    ser = open_serial("COM3", baud=115200, timeout=1.0)
    dataset = datasets.load_dataset("glue", "sst2", split="validation")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    baseline_model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ).eval()

    approx_model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ).eval()
    replace_self_attention(approx_model, BertSelfAttentionSoftmaxApprox)

    set_serial_to_model(approx_model, ser)
    correct_baseline = 0
    correct_approx = 0
    match_count_approx = 0

    lengths = []

    for i, item in enumerate(dataset):
        inputs = tokenizer(item["sentence"], return_tensors="pt", truncation=True)
        label = item["label"]

        L = inputs["input_ids"].shape[-1]
        lengths.append(L)

        with torch.no_grad():
            out_base = baseline_model(**inputs).logits
            out_approx = approx_model(**inputs).logits

        pred_base = out_base.argmax(dim=-1).item()
        pred_approx = out_approx.argmax(dim=-1).item()

        if pred_base == label:
            correct_baseline += 1
        if pred_approx == label:
            correct_approx += 1
        if pred_base == pred_approx:
            match_count_approx += 1

        same_approx = "O" if pred_base == pred_approx else "X"

        print(
            f"[{i:3d}] L={L:3d}  Base:{pred_base}  Approx:{pred_approx}  Label:{label}  Match {same_approx}"
        )

    total = len(dataset)
    print("\nEvaluation Results :")
    print(
        f"Baseline BERT Accuracy : {correct_baseline/total*100:.2f}% ({correct_baseline}/{total})"
    )
    print(
        f"Approx BERT Accuracy   : {correct_approx/total*100:.2f}% ({correct_approx}/{total})"
    )
    print(
        f"Prediction Match Rate   : {match_count_approx/total*100:.2f}% ({match_count_approx}/{total})"
    )
    close_serial(ser)


if __name__ == "__main__":
    evaluate_SST2()
