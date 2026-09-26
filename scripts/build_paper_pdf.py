"""Render docs/reports/rerank-bench-paper.md to a print-styled HTML file.

Then print to PDF with headless Chrome:
  chrome --headless=new --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="docs/reports/rerank-bench-paper.pdf" \
    "file:///<abs path>/docs/reports/rerank-bench-paper.html"
"""
from pathlib import Path

import markdown  # uv run --with markdown python scripts/build_paper_pdf.py

SRC = Path("docs/reports/rerank-bench-paper.md")
OUT = Path("docs/reports/rerank-bench-paper.html")

CSS = """
@page { size: A4; margin: 22mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; max-width: 100%; margin: 0; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 2pt; }
h1 + p { text-align: left; font-size: 10pt; color: #444; margin-top: 2pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 0.6pt solid #999;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
blockquote { border-left: 2.5pt solid #bbb; margin: 8pt 0; padding: 2pt 0 2pt 10pt;
             color: #444; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.8pt;
        font-family: 'Segoe UI', Arial, sans-serif; page-break-inside: avoid; }
th, td { border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f4;
       padding: 0 2pt; }
ol, ul { margin: 6pt 0; padding-left: 20pt; }
li { margin-bottom: 3pt; }
strong { color: #000; }
"""

md_text = SRC.read_text(encoding="utf-8")
body = markdown.markdown(md_text, extensions=["tables"])
OUT.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
               f"<title>rerank-bench technical report</title><style>{CSS}</style></head>"
               f"<body>{body}</body></html>", encoding="utf-8")
print(f"wrote {OUT}")
