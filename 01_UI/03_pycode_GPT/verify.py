# verify.py
import numpy as np
import torch

import UART_base
from VerificationGPT import (
    build_models_gpt,  # ✅
    set_serial_to_model,
    get_last_attention_matrix,
)

DEVICE = "cpu"
LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

tokenizer, baseline_model, approx_model = build_models_gpt(
    device=DEVICE, model_name="gpt2"
)


def predict_from_logits(logits_2):
    probs = torch.softmax(logits_2, dim=-1).cpu().numpy()
    pred = int(torch.argmax(logits_2).item())
    return {
        "pred_id": pred,
        "pred_label": LABEL[pred],
        "p_neg": float(probs[0]),
        "p_pos": float(probs[1]),
    }


def predict_sw_only(text: str, *, max_len: int):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
            output_attentions=False,
        )
    return predict_from_logits(out.logits[0].detach())


def compute_hw_all(
    text: str, *, layer: int, head: int, max_len: int, port: str, baud: int
):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    ser = UART_base.open_serial(port, int(baud), timeout=1.0)
    try:
        set_serial_to_model(approx_model, ser)
        with torch.no_grad():
            out = approx_model(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=inputs["attention_mask"].to(DEVICE),
                output_attentions=True,
            )
        pred = predict_from_logits(out.logits[0].detach())
        attn = np.asarray(
            get_last_attention_matrix(approx_model, layer=int(layer), head=int(head)),
            dtype=np.float64,
        )
        return tokens, attn, pred
    finally:
        try:
            ser.close()
        except Exception:
            pass
