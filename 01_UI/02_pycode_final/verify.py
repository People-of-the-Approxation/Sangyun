# verify.py
from __future__ import annotations

from typing import Optional
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
# 1. BERT Functions (그대로)
# =========================
def _load_bert():
    global _BERT_CACHE
    if _BERT_CACHE is not None:
        return _BERT_CACHE

    import UART_base
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

    pred_data = predict_sw_only(text, max_len=max_len)
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
# 2. GPT Functions (최종)
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

    # ✅ SW는 순정 모델 (UART 무관)
    sw_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()

    # ✅ HW는 attention 교체 모델
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


def run_gpt_demo(
    text: str,
    port: str,
    baud: int,
    *,
    hw_layer: int = 10,
    hw_head: int = 0,
    max_new_tokens: int = 30,
):
    """
    반환:
      - sw_text: 순정 GPT2 generate 결과 (UART 무관)
      - hw_text: "첫 1토큰만 HW" + 나머지 SW로 이어붙인 결과
                (UART 실패 시 전부 SW로 생성되지만 hw_err로 구분 가능)
      - attn_np: HW prompt forward에서 저장한 attention map (실패 시 (1,1))
      - hw_err: HW 단계 에러 메시지 (성공 시 None)
    """
    gpt = _load_gpt()
    tok = gpt["tok"]
    sw_model = gpt["sw"]
    hw_model = gpt["hw"]

    # ✅ attention_mask를 만들어 경고 제거 + 안정성 확보
    enc = tok(text, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(DEVICE)

    # -------------------------
    # 1) SW Generate (항상 가능)
    # -------------------------
    with torch.no_grad():
        sw_out = sw_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tok.eos_token_id,
        )
    sw_text = tok.decode(sw_out[0], skip_special_tokens=True)

    # -------------------------
    # 2) HW Heatmap + 1 token
    # -------------------------
    attn_np = np.zeros((1, 1), dtype=np.float64)
    hw_err: Optional[str] = None
    hw_out_ids = None

    ser = None
    try:
        # UART open 시도
        ser = gpt["UART_base"].open_serial(port, int(baud), timeout=3.0)

        # HW 모드 ON
        gpt["set_serial"](hw_model, ser)
        gpt["force_store"](hw_model, True)

        # ✅ HW 적용 레이어/헤드(heatmap target 포함)
        gpt["set_target"](
            hw_model, layer=int(hw_layer), head=int(hw_head), store_only=True
        )

        # (a) Heatmap을 얻기 위한 prompt forward
        with torch.no_grad():
            _ = hw_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # ✅ 동일 layer/head 읽기
        attn_np = gpt["get_last_attn"](hw_model, layer=int(hw_layer), head=int(hw_head))
        attn_np = np.asarray(attn_np, dtype=np.float64)

        # (b) HW로 1 token 생성
        with torch.no_grad():
            hw_out_ids = hw_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=tok.eos_token_id,
            )

    except Exception as e:
        hw_err = str(e)

    finally:
        # HW 모드 OFF + 정리
        try:
            gpt["set_target"](hw_model, 0, 0, False)
            gpt["force_store"](hw_model, False)
            gpt["clear_serial"](hw_model)
        except Exception:
            pass

        if ser:
            try:
                ser.close()
            except Exception:
                pass

    # -------------------------
    # 3) Finish Generation (SW로 마무리)
    # -------------------------
    if hw_out_ids is None:
        start_ids = input_ids
        start_mask = attention_mask
        remaining = int(max_new_tokens)
    else:
        # hw_out_ids는 "prompt + 1토큰" 길이
        start_ids = hw_out_ids
        start_mask = torch.ones_like(start_ids)  # 생성된 토큰은 전부 유효
        remaining = max(0, int(max_new_tokens) - 1)

    with torch.no_grad():
        final_out = sw_model.generate(
            input_ids=start_ids,
            attention_mask=start_mask,
            max_new_tokens=int(remaining),
            pad_token_id=tok.eos_token_id,
        )

    hw_text = tok.decode(final_out[0], skip_special_tokens=True)

    return sw_text, hw_text, attn_np, hw_err
