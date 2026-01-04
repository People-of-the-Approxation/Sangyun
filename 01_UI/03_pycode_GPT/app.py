# app.py
import io
import uuid
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse

import ui
import verify

# =========================
# Settings
# =========================
DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 256000

# id -> {"tokens": [...], "attn": np.ndarray(T,T), "meta": {...}}
ATTN_STORE = {}

app = FastAPI()


# =========================
# Pages
# =========================
@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(ui.render_root())


@app.get("/attention_ui", response_class=HTMLResponse)
def attention_ui():
    # 기본 mode를 HW로 설정
    return HTMLResponse(ui.render_attention_ui(DEFAULT_PORT, DEFAULT_BAUD))


@app.post("/attention_generate", response_class=HTMLResponse)
def attention_generate(
    text: str = Form(...),
    mode: str = Form("hw"),
    layer: int = Form(0),
    head: int = Form(0),
    max_len: int = Form(128),
    port: str = Form(DEFAULT_PORT),
    baud: int = Form(DEFAULT_BAUD),
):
    mode = (mode or "hw").lower().strip()
    if mode not in ("sw", "hw", "auto"):
        mode = "hw"

    # SW 분류는 항상 해두자 (비교용)
    pred_sw = None
    pred_sw_err = None
    try:
        pred_sw = verify.predict_sw_only(text, max_len=int(max_len))
    except Exception as e:
        pred_sw_err = str(e)

    tokens = None
    attn = None
    used_mode = mode

    pred_hw = None
    pred_hw_err = None
    auto_fallback_err = None

    # ======================
    # Heatmap source 선택
    # ======================
    if mode == "sw":
        tokens, attn, _pred = verify.compute_sw_all(
            text, layer=layer, head=head, max_len=int(max_len)
        )
        used_mode = "sw"

    elif mode == "hw":
        try:
            tokens, attn, pred_hw = verify.compute_hw_all(
                text,
                layer=layer,
                head=head,
                max_len=int(max_len),
                port=port,
                baud=int(baud),
            )
            used_mode = "hw"
        except Exception as e:
            pred_hw_err = str(e)
            # HW 강제 모드에서 HW 실패하면: 그래도 SW heatmap 보여주게 처리(사용자 경험)
            tokens, attn, _pred = verify.compute_sw_all(
                text, layer=layer, head=head, max_len=int(max_len)
            )
            used_mode = "sw"
            auto_fallback_err = pred_hw_err

    else:
        # AUTO: HW 시도 -> 실패하면 SW heatmap
        try:
            tokens, attn, pred_hw = verify.compute_hw_all(
                text,
                layer=layer,
                head=head,
                max_len=int(max_len),
                port=port,
                baud=int(baud),
            )
            used_mode = "hw"
        except Exception as e:
            auto_fallback_err = str(e)
            tokens, attn, _pred = verify.compute_sw_all(
                text, layer=layer, head=head, max_len=int(max_len)
            )
            used_mode = "sw"

    # 저장
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
            "auto_fallback_err": auto_fallback_err,
            "pred_hw_err": pred_hw_err,
            "pred_sw_err": pred_sw_err,
            "pred_sw": pred_sw,
            "pred_hw": pred_hw,
        },
    }

    T = int(attn.shape[0])

    # 비교 표시
    sw_line = "N/A"
    if pred_sw is not None:
        sw_line = f"{pred_sw['pred_label']} (Ppos={pred_sw['p_pos']:.3f}, Pneg={pred_sw['p_neg']:.3f})"

    hw_line = "N/A"
    if pred_hw is not None:
        hw_line = f"{pred_hw['pred_label']} (Ppos={pred_hw['p_pos']:.3f}, Pneg={pred_hw['p_neg']:.3f})"

    match_line = "N/A"
    if (pred_sw is not None) and (pred_hw is not None):
        match_line = "O" if pred_sw["pred_id"] == pred_hw["pred_id"] else "X"

    # 에러 블록
    err_blocks = ""
    if auto_fallback_err:
        err_blocks += f"""
        <div style='color:#c33; margin-top:8px;'>
          <b>AUTO fallback:</b> HW failed → SW used.
          <pre style="background:#f7f7f7; padding:8px; overflow:auto;">{auto_fallback_err}</pre>
        </div>
        """
    if pred_hw_err and mode == "hw":
        err_blocks += f"""
        <div style='color:#c33; margin-top:8px;'>
          <b>HW error:</b>
          <pre style="background:#f7f7f7; padding:8px; overflow:auto;">{pred_hw_err}</pre>
        </div>
        """
    if pred_sw_err:
        err_blocks += f"""
        <div style='color:#c33; margin-top:8px;'>
          <b>SW prediction error:</b>
          <pre style="background:#f7f7f7; padding:8px; overflow:auto;">{pred_sw_err}</pre>
        </div>
        """

    return HTMLResponse(
        ui.render_result_page(
            used_mode=used_mode,
            layer=int(layer),
            head=int(head),
            T=int(T),
            attn_id=attn_id,
            hw_line=hw_line,
            sw_line=sw_line,
            match_line=match_line,
            err_blocks=err_blocks,
        )
    )


@app.get("/attn_heatmap.png")
def attn_heatmap_png(id: str):
    if id not in ATTN_STORE:
        return HTMLResponse("No such attention id", status_code=404)

    tokens = ATTN_STORE[id]["tokens"]
    attn = ATTN_STORE[id]["attn"]
    meta = ATTN_STORE[id]["meta"]

    T = int(attn.shape[0])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(attn, aspect="auto")
    ax.set_title(
        f"Attention heatmap (mode={meta['mode']}, layer={meta['layer']}, head={meta['head']}, T={T})"
    )

    # 토큰 라벨은 너무 길면 깨지므로 작을 때만 표시
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
