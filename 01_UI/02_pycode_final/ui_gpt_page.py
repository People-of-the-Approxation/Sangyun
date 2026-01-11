# ui_gpt_page.py (전체 덮어쓰기)


def render_result_page(
    *,
    input_text: str,
    sw_text: str,
    hw_text: str,
    attn_id: str,  # Base64 문자열 대신 ID를 받음
    error_hw: str | None = None,
):

    err_html = ""
    if error_hw:
        err_html = f"""
        <div class="errbox">
          <div class="errtitle">HW error</div>
          <pre class="errpre">{error_hw}</pre>
        </div>
        """

    # 이미지 태그가 data:image... 가 아니라 /attn_heatmap.png?id=... 로 변경됨
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>GPT Result</title>
        <link rel="stylesheet" href="/static/style.css" />
        <style>
          :root {{ --bg: #ffffff; --text: #111; --muted: #6b7280; --line: #e5e7eb; }}
          html, body {{ height: 100%; margin: 0; background: var(--bg); color: var(--text); font-family: 'Satoshi', Arial, sans-serif; }}
          .wrap {{ display: flex; height: 100vh; width: 100vw; padding: 24px 28px; gap: 28px; box-sizing: border-box; }}
          .left {{ display: flex; align-items: center; justify-content: center; }}
          .heatmap {{ border: 1px solid var(--line); max-height: calc(100vh - 48px); max-width: 64vw; width: auto; height: auto; border-radius: 10px; }}
          .right {{ flex: 1 1 auto; min-width: 420px; display: flex; flex-direction: column; justify-content: center; position: relative; }}
          .cards {{ display: flex; flex-direction: column; gap: 14px; }}
          .card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; background: #fff; }}
          .label {{ color: var(--muted); font-size: 16px; font-weight: 400; margin-bottom: 8px; }}
          .content {{ font-size: 18px; font-weight: 400; white-space: pre-wrap; line-height: 1.35; }}
          .errbox {{ border: 1px solid #f5c2c7; background: #fff5f5; border-radius: 14px; padding: 12px 14px; }}
          .errtitle {{ color: #b42318; font-weight: 500; margin-bottom: 6px; }}
          .errpre {{ margin: 0; color: #7a1f1a; white-space: pre-wrap; font-size: 14px; }}
          .back {{ position: absolute; right: 0; bottom: 0; font-size: 28px; color: #111; text-decoration: none; }}
          .back:hover {{ text-decoration: underline; }}
        </style>
      </head>

      <body>
        <div class="wrap">
          <div class="left">
            <img class="heatmap" src="/attn_heatmap.png?id={attn_id}" />
          </div>

          <div class="right">
            <div class="cards">
              <div class="card">
                <div class="label">Input</div>
                <div class="content">{input_text}</div>
              </div>

              {err_html}

              <div class="card">
                <div class="label">SW Generated</div>
                <div class="content">{sw_text}</div>
              </div>

              <div class="card">
                <div class="label">HW Generated</div>
                <div class="content">{hw_text}</div>
              </div>
            </div>

            <a class="back" href="/attention_ui">Back</a>
          </div>
        </div>
      </body>
    </html>
    """
