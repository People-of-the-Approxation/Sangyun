# ui_main_page.py


def render_page1(
    *,
    model: str = "gpt",
    port: str = "COM3",
    mode: str = "hw",
):
    model = (model or "gpt").lower()
    mode = (mode or "hw").lower()

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Main UI</title>
        <link rel="stylesheet" href="/static/style.css" />
      </head>

      <body>
        <div class="page">

          <div class="hero-title">Chip Name blablabla</div>

          <form class="input-wrap" method="post" action="/attention_generate">

            <div class="input-frame">
              <div class="input-left">

                <div class="input-spacer"></div>

                <div class="input-label">영어 문장을 입력해주세요</div>

                <textarea
                  class="textarea"
                  name="text"
                  rows="1"
                  spellcheck="false"
                ></textarea>

                <!-- hidden 파라미터 -->
                <input type="hidden" name="layer" value="0" />
                <input type="hidden" name="head" value="0" />
                <input type="hidden" name="max_len" value="768" />
                <input type="hidden" name="baud" value="115200" />
              </div>

              <div class="input-divider"></div>

              <!-- 아이콘 = RUN -->
              <div class="input-right">
                <button type="submit" class="icon-btn" aria-label="Run">
                  <img
                    class="chip-icon"
                    src="/static/images/chip_icon.png"
                    alt="run"
                  />
                </button>
              </div>
            </div>

            <div class="control-under">
              <div class="control-row">
                <label>Model</label>
                <select name="model">
                  <option value="gpt" {"selected" if model=="gpt" else ""}>GPT</option>
                  <option value="bert" {"selected" if model=="bert" else ""}>BERT</option>
                </select>

                <label>Mode</label>
                <select name="mode">
                  <option value="hw" {"selected" if mode=="hw" else ""}>HW</option>
                  <option value="sw" {"selected" if mode=="sw" else ""}>SW</option>
                  <option value="auto" {"selected" if mode=="auto" else ""}>AUTO</option>
                </select>

                <label>Port</label>
                <input type="text" name="port" value="{port}" />
              </div>
            </div>

          </form>
        </div>
      </body>
    </html>
    """
