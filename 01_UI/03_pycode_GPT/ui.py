# ui.py


def render_root():
    return """
    <html>
      <head>
        <title>Attention Heatmap UI</title>
        <style>
          body { font-family: Arial; margin: 24px; }
          a { display:inline-block; margin-top:12px; }
        </style>
      </head>
      <body>
        <h2>Attention Heatmap UI (SST-2)</h2>
        <div><a href="/attention_ui">Open UI</a></div>
      </body>
    </html>
    """


def render_attention_ui(default_port: str, default_baud: int):
    return f"""
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
          code {{ background:#f4f4f4; padding:2px 4px; }}
        </style>
      </head>
      <body>
        <h2>Attention Heatmap</h2>

        <div class="hint">
          <b>mode</b>:
          <b>HW</b>=FPGA(UART) 기반 approx attention,
          <b>SW</b>=transformers baseline attention,
          <b>AUTO</b>=HW 먼저 시도 후 실패하면 SW fallback.
        </div>
        <div class="hint">
          결과 페이지에는 항상 <b>HW 분류</b>, <b>SW 분류</b>, <b>일치 여부</b>(O/X)를 같이 표시합니다.
        </div>

        <form method="post" action="/attention_generate">
          <div class="row">
            <input type="text" name="text" placeholder="영어 문장 입력 (예: i love you)" />
            <button type="submit">Generate</button>
          </div>

          <div class="row">
            <label>mode:</label>
            <select name="mode">
              <option value="hw" selected>HW</option>
              <option value="sw">SW</option>
              <option value="auto">AUTO</option>
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
            <input type="text" name="port" value="{default_port}" style="width:90px;" />

            <label style="margin-left:12px;">baud:</label>
            <input type="text" name="baud" value="{default_baud}" style="width:110px;" />
          </div>
        </form>
      </body>
    </html>
    """


def render_result_page(
    *,
    used_mode: str,
    layer: int,
    head: int,
    T: int,
    attn_id: str,
    hw_line: str,
    sw_line: str,
    match_line: str,
    err_blocks: str,
):
    return f"""
    <html>
      <head>
        <title>Attention Heatmap</title>
        <style>
          body {{ font-family: Arial; margin: 24px; }}
          .meta {{ color: #555; margin-top: 8px; }}
          img {{ border: 1px solid #ddd; margin-top: 16px; max-width: 1000px; }}
          a {{ display:inline-block; margin-top: 12px; }}
          .box {{ margin-top:10px; padding:10px; border:1px solid #ddd; display:inline-block; }}
          .k {{ width:110px; display:inline-block; color:#333; }}
          pre {{ margin:0; }}
        </style>
      </head>
      <body>
        <h2>Attention Heatmap</h2>
        <div class="meta">mode={used_mode}, layer={int(layer)}, head={int(head)}, T={T}</div>

        <div class="box">
          <div><span class="k"><b>HW(approx)</b></span> {hw_line}</div>
          <div><span class="k"><b>SW(baseline)</b></span> {sw_line}</div>
          <div><span class="k"><b>Match</b></span> {match_line}</div>
        </div>

        {err_blocks}

        <img src="/attn_heatmap.png?id={attn_id}" />
        <div><a href="/attention_ui">Back</a></div>
      </body>
    </html>
    """
