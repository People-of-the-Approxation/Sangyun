# app.py
# - Batch 입력(줄 단위 여러 문장)
# - BERT(멀티링구얼)로 logits 5개 생성 (한국어/영어 모두 입력 가능)
# - SW softmax(5-way) + 라벨 매핑(pos/neg/neutral)
# - FPGA UART softmax 결과(5-way) 수신 "자리" 포함 (아직 미구현이면 NOT_READY)
# - 웹에서 결과 누적 테이블 표시 + 시간/오차/매치 표시

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# =========================
# 설정
# =========================
MAX_BERT_TOKENS = 64  # tokenizer 기준 max_length (문장별)
ENABLE_FPGA = False  # FPGA 준비되면 True로
SERIAL_PORT = "COM5"  # Windows: COM5 / Linux: /dev/ttyUSB0
BAUDRATE = 115200
UART_TIMEOUT_S = 0.5

# 한국어도 되는 멀티링구얼 감정(1~5 stars) 모델
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


# =========================
# BERT 로드 (서버 시작 시 1번만)
# =========================
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def stars_label_from_probs(probs5: np.ndarray) -> Dict[str, Any]:
    """
    nlptown 모델은 클래스가 1~5점(stars) 순서라고 보는 게 일반적.
    star = argmax + 1
    매핑:
      1~2: neg
      3: neutral
      4~5: pos
    """
    star = int(np.argmax(probs5)) + 1
    if star <= 2:
        label = "neg"
    elif star == 3:
        label = "neutral"
    else:
        label = "pos"
    return {"star": star, "label": label}


def bert_logits_and_tokeninfo(sentence: str, max_tokens: int) -> Dict[str, Any]:
    """
    returns:
      logits: np.ndarray shape (5,)
      token_len_used: int (truncated 적용 후 길이)
      truncated: bool (잘렸는지)
      bert_ms: float (tokenize+forward)
    """
    s = sentence.strip()
    if not s:
        # 빈 문장은 logits 0으로
        return {
            "logits": np.zeros(5, dtype=np.float32),
            "token_len_used": 0,
            "truncated": False,
            "bert_ms": 0.0,
        }

    # truncation 여부 확인을 위해 overflowing_tokens 사용
    t0 = time.perf_counter()
    enc = tokenizer(
        s,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        return_overflowing_tokens=True,
    )
    with torch.no_grad():
        out = model(
            **{
                k: v
                for k, v in enc.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
        )
    t1 = time.perf_counter()

    logits = out.logits[0].cpu().numpy().astype(np.float32)  # (5,)
    token_len_used = int(enc["input_ids"].shape[1])
    truncated = "overflowing_tokens" in enc and enc["overflowing_tokens"].numel() > 0

    return {
        "logits": logits,
        "token_len_used": token_len_used,
        "truncated": truncated,
        "bert_ms": (t1 - t0) * 1000.0,
    }


# =========================
# FPGA UART (틀)
# =========================
@dataclass
class FpgaResult:
    probs: np.ndarray  # shape (5,)
    uart_ms: float
    raw: str


def fpga_softmax_uart_5way(logits5: np.ndarray) -> Optional[FpgaResult]:
    """
    FPGA가 준비되면 여기만 구현하면 됨.

    추천 ASCII 프로토콜 (디버깅 쉬움):
      TX:  L,5,<l0>,<l1>,<l2>,<l3>,<l4>\n
      RX:  P,5,<p0>,<p1>,<p2>,<p3>,<p4>\n

    - logits는 float로 보내도 되고, Q-format 정수로 보내도 됨(그땐 포맷 변경).
    """
    if not ENABLE_FPGA:
        return None

    import serial

    n = 5
    tx = "L,5," + ",".join(f"{float(x):.6f}" for x in logits5.tolist()) + "\n"
    txb = tx.encode("ascii")

    t0 = time.perf_counter()
    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=UART_TIMEOUT_S) as ser:
        ser.reset_input_buffer()
        ser.write(txb)
        ser.flush()
        rx = ser.readline()
    t1 = time.perf_counter()

    uart_ms = (t1 - t0) * 1000.0
    if not rx:
        return None

    raw = rx.decode("ascii", errors="replace").strip()
    try:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2 or parts[0] != "P":
            return FpgaResult(probs=np.full(n, np.nan), uart_ms=uart_ms, raw=raw)
        rn = int(parts[1])
        vals = [float(x) for x in parts[2 : 2 + rn]]
        probs = np.array(vals, dtype=np.float64)
        s = probs.sum()
        if s > 0:
            probs = probs / s
        if probs.shape[0] != 5:
            # 5개가 아니면 NaN 처리
            probs = np.full(5, np.nan)
        return FpgaResult(probs=probs, uart_ms=uart_ms, raw=raw)
    except Exception:
        return FpgaResult(probs=np.full(n, np.nan), uart_ms=uart_ms, raw=raw)


# =========================
# Web
# =========================
app = FastAPI()


class AnalyzeRequest(BaseModel):
    text: str  # 여러 문장(줄 단위)


def split_lines(text: str) -> List[str]:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n")]
    return [ln for ln in lines if ln]


@app.get("/", response_class=HTMLResponse)
def index():
    # f-string을 쓰지 않고, placeholder를 replace로 채워서
    # JS의 ${...}가 파이썬 문자열 포맷에 의해 깨지지 않게 함.
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>BERT + FPGA Softmax Verifier</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .wrap { max-width: 1200px; margin: 0 auto; }
    textarea { width: 100%; height: 140px; font-size: 15px; padding: 10px; }
    button { padding: 10px 14px; font-size: 15px; cursor: pointer; }
    .row { display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap:wrap; }
    .badge { display:inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #ccc; }
    .ok { background: #eaf7ee; border-color: #b7e2c3; }
    .warn { background: #fff6e6; border-color: #ffd28a; }
    .err { background: #ffecec; border-color: #ffb0b0; }
    .muted { color:#666; font-size:13px; }
    table { width: 100%; border-collapse: collapse; margin-top: 14px; }
    th, td { border: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #fafafa; font-weight: 700; position: sticky; top: 0; }
    .mono { font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .small { font-size: 12px; color:#555; }
    .nowrap { white-space: nowrap; }
    .pill { padding: 2px 8px; border:1px solid #ddd; border-radius: 999px; display:inline-block; }
    .pos { color: #0a7a2f; font-weight:700; }
    .neg { color: #b00020; font-weight:700; }
    .neu { color: #444; font-weight:700; }
    .scroll { overflow:auto; max-height: 560px; }
  </style>
</head>
<body>
<div class="wrap">
  <h2>BERT 감정 분류 + FPGA Softmax 비교 (5-way)</h2>
  <div class="muted">
    - 입력: 여러 문장을 <b>줄 단위</b>로 입력<br/>
    - BERT tokenizer max_length: <b>__MAXTOK__</b> (문장별)<br/>
    - FPGA: <b>__FPGAFLAG__</b> (아직 구현 중이면 OFF로 두고 틀만 맞추면 됨)<br/>
    - 모델: <span class="mono">__MODEL__</span>
  </div>

  <textarea id="text" placeholder="여러 문장을 줄 단위로 입력하세요.
예)
안녕 반가워
이 영화는 진짜 최악이야
It was okay, not great."></textarea>

  <div class="row">
    <button onclick="analyze()">Analyze (append)</button>
    <button onclick="clearHistory()">Clear history</button>
    <span id="status" class="badge">idle</span>
    <span class="pill small">Rows: <span id="count">0</span></span>
    <span class="pill small">Argmax match: <span id="matched">0</span></span>
  </div>

  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th class="nowrap">#</th>
          <th>Sentence</th>
          <th class="nowrap">Tok</th>
          <th class="nowrap">BERT ms</th>
          <th class="nowrap">SW label</th>
          <th class="nowrap">SW star</th>
          <th class="nowrap">SW probs(1..5)</th>
          <th class="nowrap">SW softmax ms</th>
          <th class="nowrap">FPGA status</th>
          <th class="nowrap">FPGA label</th>
          <th class="nowrap">FPGA star</th>
          <th class="nowrap">FPGA probs(1..5)</th>
          <th class="nowrap">FPGA ms</th>
          <th class="nowrap">Δmax</th>
          <th class="nowrap">Match</th>
          <th>Note</th>
        </tr>
      </thead>
      <tbody id="tbody">
        <tr><td colspan="16" class="muted">No results yet.</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
let historyRows = [];

function setBadge(text, cls) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "badge " + (cls || "");
}
function fmt(x, d=3) {
  if (x === null || x === undefined) return "-";
  if (!isFinite(x)) return "-";
  return Number(x).toFixed(d);
}
function fmt6(x) {
  if (x === null || x === undefined) return "-";
  if (!isFinite(x)) return "-";
  return Number(x).toFixed(6);
}
function escapeHtml(s){
  return String(s)
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
    .replaceAll('"',"&quot;").replaceAll("'","&#039;");
}
function labelClass(lbl){
  if (lbl === "pos") return "pos";
  if (lbl === "neg") return "neg";
  if (lbl === "neutral") return "neu";
  return "";
}
function render() {
  const tbody = document.getElementById("tbody");
  if (historyRows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="16" class="muted">No results yet.</td></tr>';
    document.getElementById("count").textContent = "0";
    document.getElementById("matched").textContent = "0";
    return;
  }

  let matched = 0;
  tbody.innerHTML = historyRows.map((r, i) => {
    if (r.match === "YES") matched += 1;

    const swLblCls = labelClass(r.sw_label);
    const fpLblCls = labelClass(r.fpga_label);

    const swProbs = r.sw_probs ? r.sw_probs.map(p => fmt6(p)).join(", ") : "-";
    const fpProbs = r.fpga_probs ? r.fpga_probs.map(p => fmt6(p)).join(", ") : "-";

    const noteParts = [];
    if (r.truncated) noteParts.push("truncated");
    if (r.fpga_status && r.fpga_status !== "OK") noteParts.push("FPGA:" + r.fpga_status);
    const note = noteParts.length ? noteParts.join(" / ") : "-";

    return `
      <tr>
        <td class="mono nowrap">${i+1}</td>
        <td>${escapeHtml(r.sentence)}</td>
        <td class="mono nowrap">${r.token_len_used}${r.truncated ? " (trunc)" : ""}</td>
        <td class="mono nowrap">${fmt(r.bert_ms,3)}</td>

        <td class="nowrap"><span class="${swLblCls}">${r.sw_label}</span></td>
        <td class="mono nowrap">${r.sw_star}</td>
        <td class="mono">${swProbs}</td>
        <td class="mono nowrap">${fmt(r.sw_softmax_ms,3)}</td>

        <td class="mono nowrap">${r.fpga_status}</td>
        <td class="nowrap"><span class="${fpLblCls}">${r.fpga_label ?? "-"}</span></td>
        <td class="mono nowrap">${r.fpga_star ?? "-"}</td>
        <td class="mono">${fpProbs}</td>
        <td class="mono nowrap">${r.fpga_ms === null ? "-" : fmt(r.fpga_ms,3)}</td>

        <td class="mono nowrap">${r.diffmax === null ? "-" : fmt6(r.diffmax)}</td>
        <td class="nowrap"><span class="badge ${r.match === "YES" ? "ok" : (r.match === "NO" ? "warn" : "")}">${r.match}</span></td>
        <td class="small">${note}</td>
      </tr>
    `;
  }).join("");

  document.getElementById("count").textContent = String(historyRows.length);
  document.getElementById("matched").textContent = String(matched);
}

async function analyze(){
  setBadge("running", "warn");
  const text = document.getElementById("text").value;

  const res = await fetch("/analyze_batch", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text})
  });

  const j = await res.json();
  if (j.error) {
    setBadge("error", "err");
    alert(j.error);
    return;
  }

  historyRows = historyRows.concat(j.results);
  render();

  const ok = j.summary && j.summary.processed > 0 && j.summary.matched === j.summary.processed;
  setBadge("done", ok ? "ok" : "warn");
}

function clearHistory(){
  historyRows = [];
  render();
  setBadge("idle", "");
}

render();
</script>
</body>
</html>
"""
    html = html.replace("__MAXTOK__", str(MAX_BERT_TOKENS))
    html = html.replace("__FPGAFLAG__", "ON" if ENABLE_FPGA else "OFF")
    html = html.replace("__MODEL__", MODEL_NAME)
    return HTMLResponse(content=html)


@app.post("/analyze_batch")
def analyze_batch(req: AnalyzeRequest) -> Dict[str, Any]:
    lines = split_lines(req.text)
    if not lines:
        return {"error": "입력이 비어있습니다. 줄 단위로 문장을 입력하세요."}

    results: List[Dict[str, Any]] = []
    processed = 0
    matched = 0

    for sent in lines:
        # 1) BERT logits + token info + bert time
        info = bert_logits_and_tokeninfo(sent, MAX_BERT_TOKENS)
        logits5 = info["logits"]
        token_len_used = info["token_len_used"]
        truncated = info["truncated"]
        bert_ms = info["bert_ms"]

        # 2) SW softmax (5-way) + 시간
        t0 = time.perf_counter()
        sw_probs = softmax_np(logits5)
        t1 = time.perf_counter()
        sw_softmax_ms = (t1 - t0) * 1000.0

        sw_meta = stars_label_from_probs(sw_probs)
        sw_label = sw_meta["label"]
        sw_star = sw_meta["star"]

        # 3) FPGA softmax (틀)
        fpga = fpga_softmax_uart_5way(logits5)

        fpga_status = "NOT_READY" if ENABLE_FPGA else "DISABLED"
        fpga_probs = None
        fpga_ms = None
        fpga_label = None
        fpga_star = None
        diffmax = None
        match = "N/A"

        if fpga is not None:
            fpga_ms = float(fpga.uart_ms)
            if np.any(np.isnan(fpga.probs)):
                fpga_status = "PARSE_ERROR"
            else:
                fpga_status = "OK"
                fpga_probs = fpga.probs.astype(np.float64)
                fpga_meta = stars_label_from_probs(fpga_probs)
                fpga_label = fpga_meta["label"]
                fpga_star = fpga_meta["star"]

                diffmax = float(np.max(np.abs(sw_probs - fpga_probs)))
                match = (
                    "YES"
                    if int(np.argmax(sw_probs)) == int(np.argmax(fpga_probs))
                    else "NO"
                )
                if match == "YES":
                    matched += 1

        processed += 1
        results.append(
            {
                "sentence": sent,
                "token_len_used": token_len_used,
                "truncated": truncated,
                "bert_ms": float(bert_ms),
                "sw_label": sw_label,
                "sw_star": int(sw_star),
                "sw_probs": [float(x) for x in sw_probs.tolist()],
                "sw_softmax_ms": float(sw_softmax_ms),
                "fpga_status": fpga_status,
                "fpga_label": fpga_label,
                "fpga_star": fpga_star,
                "fpga_probs": (
                    None
                    if fpga_probs is None
                    else [float(x) for x in fpga_probs.tolist()]
                ),
                "fpga_ms": fpga_ms,
                "diffmax": diffmax,
                "match": match,
            }
        )

    return {
        "results": results,
        "summary": {
            "processed": processed,
            "matched": matched,
            "fpga_enabled": ENABLE_FPGA,
            "model": MODEL_NAME,
            "max_bert_tokens": MAX_BERT_TOKENS,
        },
    }


if __name__ == "__main__":
    import uvicorn

    # 문자열 "app:app" 대신 객체 직접 넘기는 방식(모듈명 이슈 방지)
    uvicorn.run(app, host="127.0.0.1", port=8000)
