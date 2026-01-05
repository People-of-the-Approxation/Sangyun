# verify.py
"""
BERT(SST-2) + GPT 데모를 한 파일에서 제공.

app.py가 기대하는 export:
- predict_sw_only(text, max_len)
- compute_sw_all(text, layer, head, max_len)
- compute_hw_all(text, layer, head, max_len, port, baud)
- run_gpt_demo(text, port, baud) -> (sw_text, hw_text, attn_np, hw_err_or_None)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# =========================
# Common
# =========================
DEVICE = "cpu"
LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

_BERT_CACHE: Optional[Dict[str, Any]] = None
_GPT_CACHE: Optional[Dict[str, Any]] = None


# =========================
# BERT (SST-2)
# =========================
def _load_bert():
    global _BERT_CACHE
    if _BERT_CACHE is not None:
        return _BERT_CACHE

    # 프로젝트에 존재한다고 가정
    import UART_base
    from VerificationBERT import (
        build_models_sst2,
        set_serial_to_model as bert_set_serial_to_model,
        get_last_attention_matrix as bert_get_last_attention_matrix,
    )

    tokenizer, baseline_model, approx_model = build_models_sst2(device=DEVICE)

    _BERT_CACHE = {
        "UART_base": UART_base,
        "tokenizer": tokenizer,
        "baseline": baseline_model,
        "approx": approx_model,
        "set_serial": bert_set_serial_to_model,
        "get_last_attn": bert_get_last_attention_matrix,
    }
    return _BERT_CACHE


def _predict_from_logits(logits_2) -> Dict[str, Any]:
    probs = torch.softmax(logits_2, dim=-1).cpu().numpy()
    pred = int(torch.argmax(logits_2).item())
    return {
        "pred_id": pred,
        "pred_label": LABEL[pred],
        "p_neg": float(probs[0]),
        "p_pos": float(probs[1]),
    }


def predict_sw_only(text: str, *, max_len: int) -> Dict[str, Any]:
    """SW(baseline)로 분류만 수행 (attention은 안 뽑음)"""
    bert = _load_bert()
    tokenizer = bert["tokenizer"]
    baseline_model = bert["baseline"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=(
                inputs.get("attention_mask", None).to(DEVICE)
                if inputs.get("attention_mask", None) is not None
                else None
            ),
            output_attentions=False,
        )
    logits = out.logits[0].detach()
    return _predict_from_logits(logits)


def compute_sw_all(text: str, *, layer: int, head: int, max_len: int):
    """SW(baseline)로 (tokens + attention matrix + pred)"""
    bert = _load_bert()
    tokenizer = bert["tokenizer"]
    baseline_model = bert["baseline"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    with torch.no_grad():
        out = baseline_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=(
                inputs.get("attention_mask", None).to(DEVICE)
                if inputs.get("attention_mask", None) is not None
                else None
            ),
            output_attentions=True,
        )

    pred = _predict_from_logits(out.logits[0].detach())

    if out.attentions is None:
        raise RuntimeError(
            "SW model did not return attentions (output_attentions=True failed)."
        )

    L = len(out.attentions)
    layer = max(0, min(int(layer), L - 1))

    attn_l = out.attentions[layer]  # (B, heads, T, T)
    H = int(attn_l.shape[1])
    head = max(0, min(int(head), H - 1))

    attn = attn_l[0, head].detach().cpu().numpy().astype(np.float64)

    return tokens, attn, pred


def compute_hw_all(
    text: str, *, layer: int, head: int, max_len: int, port: str, baud: int
):
    """HW(approx)로 (tokens + attention matrix + pred)"""
    bert = _load_bert()
    UART_base = bert["UART_base"]
    tokenizer = bert["tokenizer"]
    approx_model = bert["approx"]
    set_serial_to_model = bert["set_serial"]
    get_last_attn = bert["get_last_attn"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    ser = UART_base.open_serial(port, int(baud), timeout=1.0)
    try:
        set_serial_to_model(approx_model, ser)
        with torch.no_grad():
            out = approx_model(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=(
                    inputs.get("attention_mask", None).to(DEVICE)
                    if inputs.get("attention_mask", None) is not None
                    else None
                ),
                output_attentions=True,
            )

        pred = _predict_from_logits(out.logits[0].detach())

        attn = get_last_attn(approx_model, layer=int(layer), head=int(head))
        if attn is None:
            raise RuntimeError(
                "HW model did not store attention. (Check output_attentions=True and custom attention last_attn.)"
            )

        attn = np.asarray(attn, dtype=np.float64)
        return tokens, attn, pred

    finally:
        try:
            ser.close()
        except Exception:
            pass


# =========================
# GPT Demo
# =========================
def _load_gpt():
    global _GPT_CACHE
    if _GPT_CACHE is not None:
        return _GPT_CACHE

    # 프로젝트에 존재한다고 가정
    import UART_base
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from VerificationGPT import (
        GPT2AttentionSoftmaxApprox,
        replace_gpt2_attention,
        set_serial_to_model as gpt_set_serial_to_model,
        clear_serial_from_model as gpt_clear_serial_from_model,
        get_last_attention_matrix as gpt_get_last_attention_matrix,
        set_force_store_attn_to_model,
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
        "set_serial": gpt_set_serial_to_model,
        "clear_serial": gpt_clear_serial_from_model,
        "get_last_attn": gpt_get_last_attention_matrix,
        "force_store": set_force_store_attn_to_model,
    }
    return _GPT_CACHE


def run_gpt_demo(*, text: str, port: str, baud: int):
    """
    반환:
      sw_text: SW 모델 생성 결과
      hw_text: HW(첫 토큰만 UART softmax) 이후 SW-local로 이어서 생성한 결과
      attn_np: heatmap용 attention (prompt 1회 forward 기준, layer=0 head=0)
      hw_err: HW 관련 에러 문자열 또는 None
    """
    gpt = _load_gpt()
    UART_base = gpt["UART_base"]
    tok = gpt["tok"]
    sw_model = gpt["sw"]
    hw_model = gpt["hw"]
    set_serial = gpt["set_serial"]
    clear_serial = gpt["clear_serial"]
    get_last_attn = gpt["get_last_attn"]
    force_store = gpt["force_store"]

    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # 1) SW generate
    with torch.no_grad():
        sw_out = sw_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    sw_text = tok.decode(sw_out[0], skip_special_tokens=True)

    # 기본값
    hw_text = ""
    hw_err: Optional[str] = None
    attn_np = np.zeros((1, 1), dtype=np.float32)

    ser = None
    try:
        # 2) HW: prompt forward로 heatmap 저장 + 첫 토큰 generate만 HW(UART)
        ser = UART_base.open_serial(port, int(baud), timeout=1.0)
        set_serial(hw_model, ser)
        force_store(hw_model, True)

        # prompt 1회 forward → (T,T) attention 저장
        with torch.no_grad():
            _ = hw_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
            )
        attn_np = np.asarray(get_last_attn(hw_model, layer=0, head=0), dtype=np.float32)

        # 첫 토큰만 HW로 생성
        with torch.no_grad():
            out1 = hw_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

    except Exception as e:
        hw_err = str(e)

        # HW 실패 시: heatmap이라도 SW-local(ser=None)로 확보 시도
        try:
            clear_serial(hw_model)
            force_store(hw_model, True)
            with torch.no_grad():
                _ = hw_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=True,
                )
            attn_np = np.asarray(
                get_last_attn(hw_model, layer=0, head=0), dtype=np.float32
            )
        except Exception:
            attn_np = np.zeros((1, 1), dtype=np.float32)

        return sw_text, hw_text, attn_np, hw_err

    finally:
        try:
            clear_serial(hw_model)
            force_store(hw_model, False)
        except Exception:
            pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass

    # 3) 나머지 토큰은 SW-local(ser=None)로 이어서 생성
    with torch.no_grad():
        out_full = hw_model.generate(
            input_ids=out1,
            max_new_tokens=29,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    hw_text = tok.decode(out_full[0], skip_special_tokens=True)

    return sw_text, hw_text, attn_np, hw_err
