# app.py
import io
import uuid
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse

import torch

from VerificationBERT import (
    build_models_sst2,
    compute_attention_sw,
    compute_attention_hw,
)

# =========================
# Settings
# =========================
DEVICE = "cpu"  # 필요하면 "cuda"로 변경

# HW default UART
DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 256000

# Attention heatmap store
# id -> {"tokens": [...], "attn": np.ndarray(T,T), "meta": {...}}
ATTN_STORE = {}

# Load models once
tokenizer, baseline_model, approx_model = build_models_sst2(device=DEVICE)

app = FastAPI()


# =========================
# Pages
# =========================
@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(
        """
    <html>
      <head>
        <title>Attention Heatmap (SST-2)</title>
        <style>
          body { font-family: Arial; margin: 24px; }
          a { display:inline-block; margin-top:12px; }
        </style>
      </head>
      <body>
        <h2>Attention Heatmap UI (textattack/bert-base-uncased-SST-2)</h2>
        <div>Go to: <a href="/attention_ui">/attention_ui</a></div>
      </body>
    </html>
    """
    )


@app.get("/attention_ui", response_class=HTMLResponse)
def attention_ui():
    return HTMLResponse(
        f"""
    <html>
      <head>
        <title>Attention Heatmap</title>
        <style>
          body {{ font-family: Arial; margin: 24px; }}
          input[type=text] {{ width: 760px; padding: 8px; }}
          button {{ padding: 8px 12px; }}
          .row {{ margin-top: 12px; }}
          .hint {{ color: #666; font-size: 13px; margin-top: 8px; }}
          select {{ padding: 6px; }}
        </style>
      </head>
      <body>
        <h2>Attention Heatmap</h2>
        <div class="hint">
          mode:
          <b>SW</b>=transformers attention,
          <b>HW</b>=FPGA(UART) softmax attention,
          <b>AUTO</b>=HW 시도 후 실패하면 SW로 fallback
        </div>

        <form method="post" action="/attention_generate">
          <div class="row">
            <input type="text" name="text" placeholder="영어 문장 입력 (SST-2 모델)" />
            <button type="submit">Generate</button>
          </div>

          <div class="row">
            <label>mode:</label>
            <select name="mode">
              <option value="auto" selected>AUTO</option>
              <option value="sw">SW</option>
              <option value="hw">HW</option>
            </select>

            <label style="margin-left:12px;">layer:</label>
            <input type="text" name="layer" value="0" style="width:50px;" />

            <label style="margin-left:12px;">head:</label>
            <input type="text" name="head" value="0" style="width:50px;" />

            <label style="margin-left:12px;">max_len:</label>
            <input type="text" name="max_len" value="128" style="width:70px;" />
          </div>

          <div class="row">
            <label>UART port:</label>
            <input type="text" name="port" value="{DEFAULT_PORT}" style="width:90px;" />

            <label style="margin-left:12px;">baud:</label>
            <input type="text" name="baud" value="{DEFAULT_BAUD}" style="width:110px;" />
          </div>
        </form>
      </body>
    </html>
    """
    )


@app.post("/attention_generate", response_class=HTMLResponse)
def attention_generate(
    text: str = Form(...),
    mode: str = Form("auto"),
    layer: int = Form(0),
    head: int = Form(0),
    max_len: int = Form(128),
    port: str = Form(DEFAULT_PORT),
    baud: int = Form(DEFAULT_BAUD),
):
    mode = (mode or "auto").lower().strip()
    if mode not in ("sw", "hw", "auto"):
        mode = "auto"

    # ---- compute attention ----
    err = None
    tokens = None
    attn = None
    used_mode = mode

    if mode == "sw":
        tokens, attn = compute_attention_sw(
            text,
            baseline_model,
            tokenizer,
            layer=layer,
            head=head,
            max_len=max_len,
            device=DEVICE,
        )
        used_mode = "sw"

    elif mode == "hw":
        tokens, attn = compute_attention_hw(
            text,
            approx_model,
            tokenizer,
            port=port,
            baud=int(baud),
            layer=layer,
            head=head,
            max_len=max_len,
            device=DEVICE,
        )
        used_mode = "hw"

    else:  # auto
        try:
            tokens, attn = compute_attention_hw(
                text,
                approx_model,
                tokenizer,
                port=port,
                baud=int(baud),
                layer=layer,
                head=head,
                max_len=max_len,
                device=DEVICE,
            )
            used_mode = "hw"
        except Exception as e:
            err = str(e)
            tokens, attn = compute_attention_sw(
                text,
                baseline_model,
                tokenizer,
                layer=layer,
                head=head,
                max_len=max_len,
                device=DEVICE,
            )
            used_mode = "sw"

    # store
    attn_id = str(uuid.uuid4())
    ATTN_STORE[attn_id] = {
        "tokens": tokens,
        "attn": np.asarray(attn, dtype=np.float64),
        "meta": {
            "mode": used_mode,
            "layer": int(layer),
            "head": int(head),
            "max_len": int(max_len),
            "port": port,
            "baud": int(baud),
            "auto_hw_error": err,
        },
    }

    T = int(attn.shape[0])
    err_html = (
        f"<div style='color:#c33; margin-top:8px;'>AUTO fallback: HW failed → SW used.<br/><pre>{err}</pre></div>"
        if err
        else ""
    )

    return HTMLResponse(
        f"""
    <html>
      <head>
        <title>Attention Heatmap</title>
        <style>
          body {{ font-family: Arial; margin: 24px; }}
          .meta {{ color: #555; margin-top: 8px; }}
          img {{ border: 1px solid #ddd; margin-top: 16px; max-width: 900px; }}
          a {{ display:inline-block; margin-top: 12px; }}
          pre {{ background: #f7f7f7; padding: 8px; overflow:auto; }}
        </style>
      </head>
      <body>
        <h2>Attention Heatmap</h2>
        <div class="meta">mode={used_mode}, layer={layer}, head={head}, T={T}</div>
        {err_html}
        <img src="/attn_heatmap.png?id={attn_id}" />
        <div><a href="/attention_ui">Back</a></div>
      </body>
    </html>
    """
    )


@app.get("/attn_heatmap.png")
def attn_heatmap_png(id: str):
    if id not in ATTN_STORE:
        return HTMLResponse("No such attention id", status_code=404)

    tokens = ATTN_STORE[id]["tokens"]
    attn = ATTN_STORE[id]["attn"]
    meta = ATTN_STORE[id]["meta"]

    T = int(attn.shape[0])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(attn, aspect="auto")
    ax.set_title(
        f"Attention heatmap (mode={meta['mode']}, layer={meta['layer']}, head={meta['head']}, T={T})"
    )

    # 토큰 라벨은 너무 길면 UI가 깨지므로, 작을 때만 표시
    if T <= 40:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
