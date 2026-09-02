import os

def render_page(filename, title, current_num, prev_file, prev_title, next_file, next_title, content_body):
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} - Transformer & LLM 交互式教材</title>
  <style>
    /* ==========================================
       1. 基础版式与全局样式
       ========================================== */
    body {{
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
      margin: 0;
      background: #f6f8fb;
      color: #1f2937;
      -webkit-font-smoothing: antialiased;
    }}

    .wrap {{
      max-width: 1160px;
      margin: 28px auto 60px;
      background: #ffffff;
      padding: 36px 44px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.07);
      border-radius: 16px;
      border: 1px solid #eef2f6;
    }}

    /* 顶部导航与元信息 */
    .nav-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid #e2e8f0;
      margin-bottom: 24px;
      font-size: 13px;
    }}

    .nav-header a {{
      color: #2563eb;
      text-decoration: none;
      font-weight: 600;
    }}

    .nav-header a:hover {{
      text-decoration: underline;
    }}

    .chapter-tag {{
      background: #2563eb;
      color: #ffffff;
      padding: 3px 8px;
      border-radius: 5px;
      font-size: 12px;
      font-weight: 700;
      margin-right: 8px;
    }}

    h1 {{
      color: #0f172a;
      font-size: 28px;
      font-weight: 800;
      margin: 0 0 12px 0;
      letter-spacing: -0.5px;
    }}

    h2 {{
      border-left: 4px solid #2563eb;
      padding-left: 12px;
      font-size: 21px;
      color: #0f172a;
      margin-top: 2.2em;
      margin-bottom: 14px;
      font-weight: 700;
    }}

    h3 {{
      font-size: 17px;
      color: #1e293b;
      margin-top: 1.5em;
      margin-bottom: 10px;
      font-weight: 600;
    }}

    p, li {{
      line-height: 1.8;
      font-size: 15px;
    }}

    ul, ol {{
      padding-left: 24px;
    }}

    li {{
      margin-bottom: 6px;
    }}

    hr {{
      border: 0;
      border-top: 1px solid #dbe3ef;
      margin: 32px 0;
    }}

    /* ==========================================
       2. 语义标注与提示框
       ========================================== */
    .mark-concept {{
      color: #2563eb;
      font-weight: 700;
    }}

    .mark-caution {{
      color: #c2410c;
      font-weight: 700;
    }}

    .mark-danger {{
      color: #dc2626;
      font-weight: 700;
    }}

    .card, .note {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 16px 20px;
      margin: 18px 0;
    }}

    .logic-box {{
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
      border-radius: 10px;
      padding: 16px 20px;
      margin: 18px 0;
    }}

    .warning-box {{
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: #9a3412;
      border-radius: 10px;
      padding: 16px 20px;
      margin: 18px 0;
    }}

    /* ==========================================
       3. 代码与表格
       ========================================== */
    code {{
      background: #eef2ff;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: Consolas, Monaco, monospace;
      font-size: 13px;
      color: #3730a3;
    }}

    pre {{
      background: #0b1020;
      color: #e2e8f0;
      padding: 16px 20px;
      border-radius: 10px;
      overflow-x: auto;
      font-size: 13.5px;
      line-height: 1.55;
      font-family: Consolas, Monaco, monospace;
      margin: 16px 0;
    }}

    pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 24px;
      font-size: 14px;
    }}

    th, td {{
      border: 1px solid #dbe3ef;
      padding: 10px 14px;
      vertical-align: middle;
    }}

    th {{
      background: #f1f5f9;
      text-align: left;
      font-weight: 600;
      color: #0f172a;
    }}

    /* ==========================================
       4. 交互式卡片与 Canvas 容器
       ========================================== */
    .interactive-card {{
      background: #ffffff;
      border: 1px solid #cfd8e3;
      border-radius: 12px;
      padding: 22px;
      margin: 22px 0;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }}

    .interactive-title {{
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .control-panel {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 14px 18px;
      margin-bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
    }}

    .control-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13.5px;
    }}

    .control-group label {{
      font-weight: 600;
      color: #334155;
    }}

    input[type="range"] {{
      accent-color: #2563eb;
      cursor: pointer;
    }}

    .btn {{
      background: #2563eb;
      color: #ffffff;
      border: none;
      padding: 7px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}

    .btn:hover {{
      background: #1d4ed8;
    }}

    .shape-badge {{
      display: inline-block;
      font-family: Consolas, monospace;
      font-size: 12px;
      background: #f1f5f9;
      color: #0f172a;
      border: 1px solid #cbd5e1;
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }}

    /* 底部导航 */
    .footer-nav {{
      display: flex;
      justify-content: space-between;
      margin-top: 40px;
      padding-top: 24px;
      border-top: 1px solid #e2e8f0;
      font-size: 14px;
    }}

    .footer-nav a {{
      color: #2563eb;
      text-decoration: none;
      font-weight: 600;
    }}

    .footer-nav a:hover {{
      text-decoration: underline;
    }}

    footer {{
      text-align: center;
      margin-top: 30px;
      color: #94a3b8;
      font-size: 12.5px;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
  <main class="wrap">
    <div class="nav-header">
      <div>
        <span class="chapter-tag">第 {current_num} 节</span>
        <a href="index.html">← 返回目录</a>
      </div>
      <div>
        <a href="{prev_file}">← {prev_title}</a> | 
        <a href="{next_file}">{next_title} →</a>
      </div>
    </div>

    {content_body}

    <div class="footer-nav">
      <a href="{prev_file}">← 上一节：{prev_title}</a>
      <a href="index.html">📑 目录主页</a>
      <a href="{next_file}">下一节：{next_title} →</a>
    </div>

    <footer>
      Transformer & LLM 底层原理与神经网络交互式教材 &copy; 2026
    </footer>
  </main>
</body>
</html>"""
    path = os.path.join("/mnt/c/myprog/ai/TransformerBook", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated successfully: {filename}")
