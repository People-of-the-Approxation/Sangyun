# verify.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch

# =========================
# Common Settings
# =========================
DEVICE = "cpu"
LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

_BERT_CACHE = None
_GPT_CACHE = None


# =========================
# 1. BERT Functions
# =========================
def _load_bert():
    global _BERT_CACHE
    if _BERT_CACHE is not None:
        return _BERT_CACHE

    import UART_base

    # VerificationBERT가 복구되었는지 확인하세요
    from VerificationBERT import (
        build_models_sst2,
        set_serial_to_model,
        get_last_attention_matrix,
    )

    tokenizer, baseline, approx = build_models_sst2(device=DEVICE)
    _BERT_CACHE = {
        "UART_base": UART_base,
        "tokenizer": tokenizer,
        "baseline": baseline,
        "approx": approx,
        "set_serial": set_serial_to_model,
        "get_last_attn": get_last_attention_matrix,
    }
    return _BERT_CACHE


def predict_sw_only(text: str, *, max_len: int):
    bert = _load_bert()
    inputs = bert["tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=max_len
    )
    with torch.no_grad():
        out = bert["baseline"](input_ids=inputs["input_ids"].to(DEVICE))

    probs = torch.softmax(out.logits[0], dim=-1).cpu().numpy()
    pred = int(torch.argmax(out.logits[0]).item())
    return {
        "pred_id": pred,
        "pred_label": LABEL[pred],
        "p_neg": float(probs[0]),
        "p_pos": float(probs[1]),
    }


def compute_sw_all(text: str, *, layer: int, head: int, max_len: int):
    bert = _load_bert()
    inputs = bert["tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=max_len
    )
    ids = inputs["input_ids"][0].tolist()
    tokens = bert["tokenizer"].convert_ids_to_tokens(ids)

    with torch.no_grad():
        out = bert["baseline"](
            input_ids=inputs["input_ids"].to(DEVICE), output_attentions=True
        )

    pred_data = predict_sw_only(text, max_len=max_len)  # 재사용하거나 logits 파싱

    attn = out.attentions[layer][0, head].detach().cpu().numpy().astype(np.float64)
    return tokens, attn, pred_data


def compute_hw_all(
    text: str, *, layer: int, head: int, max_len: int, port: str, baud: int
):
    bert = _load_bert()
    inputs = bert["tokenizer"](
        text, return_tensors="pt", truncation=True, max_length=max_len
    )
    ids = inputs["input_ids"][0].tolist()
    tokens = bert["tokenizer"].convert_ids_to_tokens(ids)

    ser = bert["UART_base"].open_serial(port, int(baud), timeout=1.0)
    try:
        bert["set_serial"](bert["approx"], ser)
        with torch.no_grad():
            out = bert["approx"](
                input_ids=inputs["input_ids"].to(DEVICE), output_attentions=True
            )

        probs = torch.softmax(out.logits[0], dim=-1).cpu().numpy()
        pred = int(torch.argmax(out.logits[0]).item())
        pred_data = {
            "pred_id": pred,
            "pred_label": LABEL[pred],
            "p_neg": float(probs[0]),
            "p_pos": float(probs[1]),
        }

        attn = bert["get_last_attn"](bert["approx"], layer=layer, head=head)
        return tokens, np.asarray(attn, dtype=np.float64), pred_data
    finally:
        if ser:
            ser.close()


# =========================
# 2. GPT Functions
# =========================
def _load_gpt():
    global _GPT_CACHE
    if _GPT_CACHE is not None:
        return _GPT_CACHE

    import UART_base
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from VerificationGPT import (
        GPT2AttentionSoftmaxApprox,
        replace_gpt2_attention,
        set_serial_to_model,
        clear_serial_from_model,
        get_last_attention_matrix,
        set_force_store_attn_to_model,
        set_store_target_to_model,
    )

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sw_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    hw_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    replace_gpt2_attention(hw_model, GPT2AttentionSoftmaxApprox)

    _GPT_CACHE = {
        "UART_base": UART_base,
        "tok": tok,
        "sw": sw_model,
        "hw": hw_model,
        "set_serial": set_serial_to_model,
        "clear_serial": clear_serial_from_model,
        "get_last_attn": get_last_attention_matrix,
        "force_store": set_force_store_attn_to_model,
        "set_target": set_store_target_to_model,
    }
    return _GPT_CACHE


def run_gpt_demo(text: str, port: str, baud: int):
    gpt = _load_gpt()
    tok = gpt["tok"]
    hw_model = gpt["hw"]

    inputs = tok(text, return_tensors="pt").to(DEVICE)

    # 1. SW Generate (Pure SW model use for speed reference, or use hw_model without serial)
    # 여기서는 빠른 응답을 위해 hw_model(Serial=None)을 SW로 사용
    gpt["clear_serial"](hw_model)
    with torch.no_grad():
        sw_out = hw_model.generate(
            **inputs, max_new_tokens=30, pad_token_id=tok.eos_token_id
        )
    sw_text = tok.decode(sw_out[0], skip_special_tokens=True)

    # 2. HW Heatmap & First Token
    hw_text = ""
    hw_err = None
    attn_np = np.zeros((1, 1), dtype=np.float32)

    ser = None
    try:
        ser = gpt["UART_base"].open_serial(
            port, int(baud), timeout=3.0
        )  # Timeout 넉넉히
        gpt["set_serial"](hw_model, ser)
        gpt["force_store"](hw_model, True)
        gpt["set_target"](hw_model, layer=0, head=0, store_only=True)

        # Heatmap Extraction (Prompt Forward)
        with torch.no_grad():
            _ = hw_model(**inputs, output_attentions=True)
        attn_np = gpt["get_last_attn"](hw_model, layer=0, head=0)

        # Generate 1 token with HW
        with torch.no_grad():
            hw_out_ids = hw_model.generate(
                **inputs, max_new_tokens=1, pad_token_id=tok.eos_token_id
            )

    except Exception as e:
        hw_err = str(e)
    finally:
        # Serial Cleanup
        gpt["set_target"](hw_model, 0, 0, False)
        gpt["force_store"](hw_model, False)
        gpt["clear_serial"](hw_model)
        if ser:
            ser.close()

    # 3. Finish Generation (SW mode)
    # HW 에러가 났든 안 났든, 남은 토큰은 SW로 빠르게 생성
    if "hw_out_ids" not in locals():
        hw_out_ids = inputs["input_ids"]

    with torch.no_grad():
        final_out = hw_model.generate(
            input_ids=hw_out_ids, max_new_tokens=29, pad_token_id=tok.eos_token_id
        )

    hw_text = tok.decode(final_out[0], skip_special_tokens=True)

    return sw_text, hw_text, attn_np, hw_err
