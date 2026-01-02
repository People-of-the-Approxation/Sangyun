from typing import Optional, Tuple
import torch
import datasets
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention
from Attention_approx import attention
import UART_base

PORT = "COM6"
BAUD = 256000

ser = UART_base.open_serial(PORT, BAUD)


class BertSelfAttentionSoftmaxApprox(BertSelfAttention):
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

        for b in range(B):
            for h in range(H):
                Q_np = query_layer[b, h].detach().cpu().numpy()
                K_np = key_layer[b, h].detach().cpu().numpy()
                V_np = value_layer[b, h].detach().cpu().numpy()
                if mask_np is not None:
                    Vm = V_np.copy()
                    Vm[mask_np[b] < 0] = 0
                    out_np = attention(Q_np, K_np, V_np, ser)
                else:
                    out_np = attention(Q_np, K_np, V_np, ser)
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


def evaluate_SST2():
    dataset = datasets.load_dataset("glue", "sst2", split="validation")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    baseline_model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ).eval()

    approx_model = BertForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ).eval()
    replace_self_attention(approx_model, BertSelfAttentionSoftmaxApprox)

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


if __name__ == "__main__":
    evaluate_SST2()
