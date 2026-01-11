# app.py
import io
import uuid
import asyncio
import contextlib
import numpy as np
import torch
import matplotlib

# 서버 환경에서 GUI 창이 뜨지 않도록 설정
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# --- 기존 파일들 임포트 ---
import ui_main_page
import ui_gpt_page as ui_gpt
import ui_bert_page
from softmax_batch import open_serial, close_serial

# 파일명 불일치 해결 및 GPT Attention 추출 함수 임포트
from VerificationBERT import build_model_BERT, get_last_attention_matrix
from VerificationGPT2 import build_model_GPT2, get_last_gpt2_attention_matrix

# =========================
# Settings & Global State
# =========================
SERIAL_PORT = "COM3"  # 환경에 맞게 변경
BAUD_RATE = 115200
TIMEOUT = 1.0

# 전역 변수 저장소
models = {}
ATTN_STORE = {}  # BERT/GPT 히트맵 저장용
hardware_lock = asyncio.Lock()  # 시리얼 포트 충돌 방지


# =========================
# Lifespan (서버 시작/종료 시 실행)
# =========================
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. 시리얼 연결 시도 (실패해도 죽지 않음)
    print(f"[System] Attempting to open serial port {SERIAL_PORT}...")
    ser = None
    try:
        ser = open_serial(SERIAL_PORT, baud=BAUD_RATE, timeout=TIMEOUT)
        print(f"[System] Serial port {SERIAL_PORT} connected successfully.")
    except Exception as e:
        print(f"[Warning] Failed to open serial port: {e}")
        print("[System] Starting in SOFTWARE ONLY mode.")
        ser = None

    # 2. 모델 빌드 (시리얼 연결 여부와 상관없이 무조건 실행)
    print("[System] Building BERT model...")
    tok_bert, base_bert, approx_bert, dev_bert = build_model_BERT(ser)

    print("[System] Building GPT-2 model...")
    tok_gpt2, base_gpt2, approx_gpt2, dev_gpt2 = build_model_GPT2(ser)

    # 3. 전역 변수에 등록
    models["ser"] = ser
    models["bert"] = (tok_bert, base_bert, approx_bert, dev_bert)
    models["gpt2"] = (tok_gpt2, base_gpt2, approx_gpt2, dev_gpt2)

    print("[System] All models loaded and ready!")

    yield  # 서버 실행 중

    # 4. 종료 시: 시리얼 닫기
    if models.get("ser"):
        print("[System] Closing serial port...")
        close_serial(models["ser"])


app = FastAPI(lifespan=lifespan)

# static 폴더 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# Routes: Pages
# =========================
@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/attention_ui")


@app.get("/attention_ui", response_class=HTMLResponse)
async def attention_ui(
    model: str = "gpt",
    port: str = SERIAL_PORT,
    mode: str = "hw",
):
    return HTMLResponse(
        ui_main_page.render_page1(
            model=model,
            port=port,
            mode=mode,
        )
    )


# =========================
# Routes: Generation Logic
# =========================
@app.post("/attention_generate", response_class=HTMLResponse)
async def attention_generate(
    text: str = Form(...),
    model: str = Form("gpt"),
    mode: str = Form("hw"),
    layer: int = Form(0),
    head: int = Form(0),
    max_len: int = Form(128),
    port: str = Form(SERIAL_PORT),
    baud: int = Form(BAUD_RATE),
):
    if "bert" not in models and "gpt2" not in models:
        return HTMLResponse(
            "<h1>Error: Models initialization failed completely. Check logs.</h1>"
        )

    async with hardware_lock:
        if model == "bert":
            return process_bert(text, mode, layer, head, max_len)
        elif model == "gpt":
            return process_gpt(text, mode, layer, head)
        else:
            return HTMLResponse(f"<h1>Error: Unknown model '{model}'</h1>")


# =========================
# Logic: BERT
# =========================
def process_bert(text, mode, layer, head, max_len):
    tokenizer, base_model, approx_model, device = models["bert"]

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_len
    ).to(device)

    # Baseline (SW) - 히트맵 추출을 위해 output_attentions=True
    try:
        with torch.no_grad():
            out_base = base_model(**inputs, output_attentions=True)
        sw_probs = torch.softmax(out_base.logits, dim=-1).tolist()[0]
        pred_sw_idx = out_base.logits.argmax().item()
    except Exception as e:
        return HTMLResponse(f"<h1>SW Error: {e}</h1>")

    # Hardware (Approx)
    hw_probs = [0.0, 0.0]
    pred_hw_idx = -1
    hw_error = None

    if mode in ["hw", "auto"]:
        try:
            with torch.no_grad():
                out_approx = approx_model(**inputs).logits
            hw_probs = torch.softmax(out_approx, dim=-1).tolist()[0]
            pred_hw_idx = out_approx.argmax().item()
        except Exception as e:
            hw_error = str(e)

    # 2. Attention Matrix 추출
    attn_matrix = None

    # A. HW가 성공했으면 HW 히트맵 사용
    if mode in ["hw", "auto"] and hw_error is None:
        attn_matrix = get_last_attention_matrix(approx_model, layer=layer, head=head)

    # B. HW가 실패했거나 SW모드라면 -> SW 모델(Baseline)에서 히트맵 가져오기
    if attn_matrix is None and out_base.attentions is not None:
        try:
            # Baseline의 특정 레이어/헤드 Attention 가져오기
            sw_attn_layer = out_base.attentions[int(layer)]
            attn_matrix = sw_attn_layer[0, int(head), :, :].cpu().numpy()

            if hw_error:
                hw_error += "\n(Displaying SW Heatmap instead)"
            else:
                hw_error = "(Mode: SW - Displaying Baseline Heatmap)"
        except IndexError:
            pass

    if attn_matrix is None:
        T = inputs["input_ids"].shape[-1]
        attn_matrix = np.zeros((T, T))

    # 3. Store Heatmap
    attn_id = str(uuid.uuid4())
    token_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    ATTN_STORE[attn_id] = {
        "attn": attn_matrix,
        "tokens": tokens,
        "meta": {"mode": mode, "layer": layer, "head": head},
    }

    match_line = "N/A"
    if pred_hw_idx != -1:
        match_line = "MATCH" if pred_sw_idx == pred_hw_idx else "MISMATCH"

    return HTMLResponse(
        ui_bert_page.render_bert_result_page(
            used_mode=mode,
            layer=layer,
            head=head,
            T=len(tokens),
            attn_id=attn_id,
            hw_ppos=hw_probs[1],
            hw_pneg=hw_probs[0],
            sw_ppos=sw_probs[1],
            sw_pneg=sw_probs[0],
            match_line=match_line,
            err_blocks=hw_error if hw_error else "",
        )
    )


# =========================
# Logic: GPT (수정됨: SW 히트맵 폴백 추가)
# =========================
def process_gpt(text, mode, layer, head):
    tokenizer, base_model, approx_model, device = models["gpt2"]

    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # 1. Baseline (SW)
    # SW 히트맵 추출을 위해 output_attentions=True, return_dict_in_generate=True 설정
    out_base = base_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        output_attentions=True,  # Attention 출력 활성화
        return_dict_in_generate=True,  # 딕셔너리 형태로 반환
    )
    sw_text = tokenizer.decode(out_base.sequences[0], skip_special_tokens=True)

    # 2. Hardware (Approx)
    hw_text = ""
    hw_error = None
    real_attn = None
    out_approx = None

    if mode in ["hw", "auto"]:
        try:
            out_approx = approx_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
            hw_text = tokenizer.decode(out_approx[0], skip_special_tokens=True)

            # HW Attention 가져오기
            real_attn = get_last_gpt2_attention_matrix(
                approx_model, layer=layer, head=head
            )

        except Exception as e:
            hw_error = str(e)
            hw_text = "HW Generation Failed"

    # 3. Fallback: HW 실패 or SW모드 -> Baseline에서 Attention 가져오기
    if real_attn is None and out_base.attentions is not None:
        try:
            # out_base.attentions는 (step, layer) 튜플 구조
            # 가장 마지막 생성 단계(-1)의 특정 레이어 Attention 추출
            last_step_layers = out_base.attentions[-1]
            target_layer_attn = last_step_layers[int(layer)]  # (Batch, Head, T, T)

            # (Batch=0, Head=target)
            real_attn = target_layer_attn[0, int(head), :, :].cpu().numpy()

            if hw_error:
                hw_error += "\n(Displaying SW Heatmap instead)"
            else:
                hw_error = "(Mode: SW - Displaying Baseline Heatmap)"
        except Exception as e:
            pass

    # 4. 히트맵 데이터 저장
    attn_id = str(uuid.uuid4())

    # 토큰 라벨 설정 (HW가 성공했으면 HW 토큰, 아니면 SW 토큰)
    final_tokens_seq = (
        out_approx[0] if (out_approx is not None) else out_base.sequences[0]
    )
    tokens = [tokenizer.decode([t]).strip() for t in final_tokens_seq]

    # 그래도 히트맵이 없으면 빈 행렬
    if real_attn is None:
        real_attn = np.zeros((10, 10))
        tokens = ["Err"] * 10

    ATTN_STORE[attn_id] = {
        "attn": real_attn,
        "tokens": tokens,
        "meta": {"mode": mode, "layer": layer, "head": head, "model": "GPT"},
    }

    return HTMLResponse(
        ui_gpt.render_result_page(
            input_text=text,
            sw_text=sw_text,
            hw_text=hw_text,
            attn_id=attn_id,
            error_hw=hw_error,
        )
    )


# =========================
# Image Serving
# =========================
@app.get("/attn_heatmap.png")
async def attn_heatmap_png(id: str):
    # 에러 발생 시 텍스트 이미지를 반환하는 함수 (디버깅용)
    def error_image(msg):
        fig_err, ax_err = plt.subplots(figsize=(5, 1))
        ax_err.text(0.5, 0.5, msg, ha="center", va="center", color="red")
        ax_err.axis("off")
        buf_err = io.BytesIO()
        fig_err.savefig(buf_err, format="png")
        plt.close(fig_err)
        buf_err.seek(0)
        return StreamingResponse(buf_err, media_type="image/png")

    if id not in ATTN_STORE:
        return error_image("Image not found in STORE")

    try:
        data = ATTN_STORE[id]
        attn = data["attn"]
        tokens = data["tokens"]
        meta = data["meta"]

        T_rows, T_cols = attn.shape
        L = len(tokens)

        # [핵심 수정] 히트맵 크기(T_cols)와 토큰 개수(L)가 다를 경우 강제 보정
        # GPT-2 생성 시 (Attention 크기) = (토큰 개수 - 1)인 경우가 많음
        display_tokens = tokens
        if L != T_cols:
            # 토큰이 더 많으면 뒤를 자름 (예: 15개 -> 14개)
            if L > T_cols:
                display_tokens = tokens[:T_cols]
            # 토큰이 모자라면 빈 칸 채움
            else:
                display_tokens = tokens + [""] * (T_cols - L)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # 컬러 팔레트 설정 (BERT/GPT 공통)
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
        light_colors = [mcolors.to_hex(light_cmap(i / 6)) for i in range(7)]
        dark_colors = base_colors[5:]
        colors_12 = light_colors + dark_colors
        bounds = np.linspace(0.0, 1.0, len(colors_12) + 1)
        cmap = ListedColormap(colors_12)
        norm = BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(attn, aspect="auto", cmap=cmap, norm=norm)

        model_name = meta.get("model", "Model")
        ax.set_title(f"{model_name} Attention (L{meta['layer']} H{meta['head']})")

        # 축 라벨 설정 (토큰 개수가 64개 이하일 때만 표시)
        if T_cols <= 64:
            # 여기서 tokens 대신 보정된 display_tokens를 사용해야 에러가 안 남
            ax.set_xticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)

            # 행(Query) 축 라벨
            if T_rows == len(display_tokens):
                ax.set_yticks(range(len(display_tokens)))
                ax.set_yticklabels(display_tokens, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"[Heatmap Generation Error] {e}")
        # 에러가 나면 콘솔뿐 아니라 이미지로도 에러 내용을 보여줌
        return error_image(f"Plot Error: {str(e)[:50]}...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
