# app.py
import io
import uuid

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import ui_main_page
import ui_gpt_page as ui
import ui_bert_page  # ✅ BERT 결과 페이지(ui_bert_page.py)
import verify

# =========================
# Settings
# =========================
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 115200

# BERT heatmap store (id -> dict)
ATTN_STORE = {}

app = FastAPI()

# static 폴더 서빙 (CSS/이미지)
# project/static/style.css
# project/static/images/chip_icon.png
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# UI Page (ONLY ONE)
# =========================
@app.get("/", response_class=HTMLResponse)
def root():
    # 별도 main page 제거: 기존 UI로 바로 보내기
    return RedirectResponse(url="/attention_ui")


@app.get("/attention_ui", response_class=HTMLResponse)
def attention_ui(
    model: str = "gpt",
    port: str = DEFAULT_PORT,
    mode: str = "hw",
):
    # 기존 UI 그대로: model 토글 포함
    return HTMLResponse(
        ui_main_page.render_page1(
            model=model,
            port=port,
            mode=mode,
        )
    )


# =========================
# Helpers
# =========================
def _run_gpt_demo(text: str, port: str, baud: int):
    """
    verify.py의 GPT 데모 함수 호출.
    기대 반환: (sw_text, hw_text, attn_np, hw_error_str_or_None)
    """
    candidates = [
        "run_gpt_demo",
        "gpt_demo",
        "compute_gpt_all",
        "compute_gpt_demo",
    ]
    for fn_name in candidates:
        if hasattr(verify, fn_name):
            fn = getattr(verify, fn_name)
            return fn(text=text, port=port, baud=baud)

    raise RuntimeError(
        "GPT mode selected, but verify.py does not provide a GPT demo function.\n"
        "Expected one of: run_gpt_demo / gpt_demo / compute_gpt_all / compute_gpt_demo\n"
        "Please add a function that returns (sw_text, hw_text, attn_np, hw_error_or_None)."
    )


# =========================
# Generate (Model-dependent Result Page)
# =========================
@app.post("/attention_generate", response_class=HTMLResponse)
def attention_generate(
    text: str = Form(...),
    model: str = Form("gpt"),
    mode: str = Form("hw"),
    layer: int = Form(0),
    head: int = Form(0),
    max_len: int = Form(128),
    port: str = Form(DEFAULT_PORT),
    baud: int = Form(DEFAULT_BAUD),
):
    model = (model or "gpt").lower().strip()

    # -----------------------------
    # 1) GPT: generation result page (ui.py)
    # -----------------------------
    if model == "gpt":
        try:
            sw_text, hw_text, attn_np, hw_err = _run_gpt_demo(
                text=text, port=port, baud=int(baud)
            )
        except Exception as e:
            sw_text, hw_text = "", ""
            attn_np = np.zeros((1, 1), dtype=np.float32)
            hw_err = str(e)

        heatmap_b64 = ui._attn_to_png_base64(np.asarray(attn_np, dtype=np.float32))

        return HTMLResponse(
            ui.render_result_page(
                input_text=text,
                sw_text=sw_text,
                hw_text=hw_text,
                heatmap_png_b64=heatmap_b64,
                error_hw=hw_err,
            )
        )

    # -----------------------------
    # 2) BERT: classification + match + heatmap page
    # -----------------------------
    mode = (mode or "hw").lower().strip()
    if mode not in ("sw", "hw", "auto"):
        mode = "hw"

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
            # HW 강제 모드에서 HW 실패하면 SW heatmap이라도 보여주기
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

    attn = np.asarray(attn, dtype=np.float64)
    attn_id = str(uuid.uuid4())
    ATTN_STORE[attn_id] = {
        "tokens": tokens,
        "attn": attn,
        "meta": {
            "mode": used_mode,
            "layer": int(layer),
            "head": int(head),
            "max_len": int(max_len),
            "port": port,
            "baud": int(baud),
        },
    }

    T = int(attn.shape[0])

    sw_line = "N/A"
    if pred_sw is not None:
        sw_line = (
            f"{pred_sw['pred_label']} "
            f"(Ppos={pred_sw['p_pos']:.3f}, Pneg={pred_sw['p_neg']:.3f})"
        )

    hw_line = "N/A"
    if pred_hw is not None:
        hw_line = (
            f"{pred_hw['pred_label']} "
            f"(Ppos={pred_hw['p_pos']:.3f}, Pneg={pred_hw['p_neg']:.3f})"
        )

    match_line = "N/A"
    if (pred_sw is not None) and (pred_hw is not None):
        match_line = "O" if pred_sw["pred_id"] == pred_hw["pred_id"] else "X"

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
        ui_bert_page.render_bert_result_page(
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


# =========================
# BERT Heatmap Image
# =========================
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

    # 12-step palette
    base_colors = [
        "#F6EAE8",
        "#F2CEBE",
        "#ECAE96",
        "#E7916F",
        "#E47950",
        "#E06338",
        "#D85D34",
        "#CB552E",
        "#BD4E2A",
        "#A84121",
    ]

    light_cmap = LinearSegmentedColormap.from_list("light_part", base_colors[:5])
    light_colors = [mcolors.to_hex(light_cmap(i / 6)) for i in range(7)]  # 7
    dark_colors = base_colors[5:]  # 5
    colors_12 = light_colors + dark_colors

    bounds = np.linspace(0.0, 1.0, len(colors_12) + 1)
    cmap = ListedColormap(colors_12)
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(attn, aspect="auto", cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        f"BERT Attention (mode={meta['mode']}, layer={meta['layer']}, head={meta['head']}, T={T})"
    )

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
