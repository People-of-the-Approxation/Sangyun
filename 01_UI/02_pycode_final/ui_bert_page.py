# ui_bert_page.py


def render_bert_result_page(
    *,
    used_mode: str,
    layer: int,
    head: int,
    T: int,
    attn_id: str,
    hw_line: str,
    sw_line: str,
    match_line: str,
    err_blocks: str = "",
):
    return f"""
    <html>
      <head>
        <title>BERT Attention</title>
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
        <h2>BERT Attention Heatmap</h2>
        <div class="meta">mode={used_mode}, layer={int(layer)}, head={int(head)}, T={int(T)}</div>

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
