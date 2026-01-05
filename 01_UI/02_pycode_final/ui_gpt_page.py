# ui_gpt_page.py
import numpy as np  # ✅ 이 줄 추가
from io import BytesIO
import base64
import matplotlib.pyplot as plt


def _attn_to_png_base64(attn: np.ndarray) -> str:
    # attn: (T,T)
    fig = plt.figure()
    plt.imshow(attn, vmin=attn.min(), vmax=attn.max())
    plt.colorbar()
    plt.title("Attention Heatmap (HW hybrid: row0=HW, others=SW)")
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return data


def render_attention_ui(default_port: str):
    return f"""
    <html>
      <head>
        <title>GPT Softmax HW Demo</title>
        <style>
          body {{ font-family: Arial; margin: 24px; }}
          .row {{ margin-top: 12px; }}
          input[type=text] {{ padding: 8px; }}
          #text {{ width: 860px; }}
          button {{ padding: 8px 12px; }}
          .hint {{ color:#666; font-size:13px; margin-top:8px; }}
          code {{ background:#f4f4f4; padding:2px 4px; }}
        </style>
      </head>
      <body>
        <h2>GPT Demo (SW generate + HW generate + Heatmap)</h2>
        <div class="hint">
          고정 설정: <code>baud=115200</code>, <code>layer=0</code>, <code>head=0</code>, <code>max_len=768</code><br/>
          Heatmap은 prompt 1회 forward 기준이며, softmax는 <b>첫 row만 HW</b>, 나머지는 <b>SW local</b>로 계산됨.
        </div>

        <form method="post" action="/attention_generate">
          <div class="row">
            <input id="text" type="text" name="text" placeholder="Enter an English sentence..." />
            <button type="submit">Run</button>
          </div>

          <div class="row">
            <label>UART port:</label>
            <input type="text" name="port" value="{default_port}" style="width:110px;" />
          </div>
        </form>
      </body>
    </html>
    """


def render_result_page(
    *,
    input_text: str,
    sw_text: str,
    hw_text: str,
    heatmap_png_b64: str,
    error_hw: str | None = None,
):
    err_html = ""
    if error_hw:
        err_html = f"""
        <div style="margin-top:12px; padding:10px; background:#fff3cd; border:1px solid #ffeeba;">
          <b>HW error:</b> {error_hw}
        </div>
        """

    return f"""
    <html>
      <head>
        <title>Result</title>
        <style>
          body {{ font-family: Arial; margin: 24px; }}
          .card {{ border:1px solid #ddd; border-radius:10px; padding:14px; margin-top:12px; }}
          .label {{ color:#666; font-size:12px; }}
          .mono {{ font-family: Consolas, monospace; white-space: pre-wrap; }}
          img {{ max-width: 1200px; width: 100%; border:1px solid #ddd; border-radius:10px; }}
          a {{ text-decoration:none; }}
        </style>
      </head>
      <body>
        <a href="/attention_ui">← Back</a>
        <h2>Result</h2>

        <div class="card">
          <div class="label">Input</div>
          <div class="mono">{input_text}</div>
        </div>

        {err_html}

        <div class="card">
          <div class="label">SW Generated</div>
          <div class="mono">{sw_text}</div>
        </div>

        <div class="card">
          <div class="label">HW Generated (softmax row0 via UART)</div>
          <div class="mono">{hw_text}</div>
        </div>

        <div class="card">
          <div class="label">Heatmap</div>
          <img src="data:image/png;base64,{heatmap_png_b64}" />
        </div>
      </body>
    </html>
    """
