"""Render modeldna-tech-report.md to PDF via Microsoft Edge headless.

Usage: python paper/make_pdf.py
Needs: pip install markdown, and Edge (or set MODELDNA_BROWSER to a
Chromium-family binary that supports --headless --print-to-pdf).
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).parent
SRC = HERE / "modeldna-tech-report.md"
PNG = HERE / "fig_slerp_tcurves.png"
PDF_OUT = HERE / "modeldna-tech-report.pdf"
BROWSER = os.environ.get(
    "MODELDNA_BROWSER",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)

CSS = """
@page { size: A4; margin: 22mm 20mm; }
* { box-sizing: border-box; }
html { font-size: 10.6pt; }
body {
  font-family: Cambria, Georgia, 'Times New Roman', serif;
  color: #111; line-height: 1.5; max-width: 165mm; margin: 0 auto;
}
h1 { font-size: 1.65rem; line-height: 1.25; margin: 0 0 0.4rem; }
h1 + p { margin-top: 0.2rem; }
h2 { font-size: 1.22rem; margin: 1.6rem 0 0.5rem; page-break-after: avoid; }
h3 { font-size: 1.05rem; margin: 1.2rem 0 0.4rem; page-break-after: avoid; }
p { margin: 0.55rem 0; text-align: justify; hyphens: auto; }
body > p:nth-of-type(-n+2) { text-align: left; }  /* author + links block */
li { margin: 0.25rem 0; }
strong { font-weight: 600; }
a { color: #1a4f8f; text-decoration: none; }
hr { border: 0; border-top: 0.5pt solid #999; margin: 1.2rem 0; }
code {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 0.88em; background: #f4f3f0; padding: 0 2px; border-radius: 2px;
}
pre {
  font-family: Consolas, 'Courier New', monospace; font-size: 0.85em;
  background: #f4f3f0; padding: 8px 10px; border-radius: 3px;
  overflow-x: hidden; white-space: pre-wrap; page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
table {
  border-collapse: collapse; margin: 0.8rem auto; font-size: 0.92em;
  page-break-inside: avoid;
}
th, td { border: 0.5pt solid #888; padding: 3px 9px; text-align: left; }
th { background: #f0efec; font-weight: 600; }
img { max-width: 100%; display: block; margin: 1rem auto 0.3rem; page-break-inside: avoid; }
p:has(> img) + p { text-align: left; font-size: 0.9em; color: #333; }
blockquote { margin: 0.6rem 1.2rem; color: #333; }
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    uri = "data:image/png;base64," + base64.b64encode(PNG.read_bytes()).decode()
    text = text.replace("(fig_slerp_tcurves.png)", "(" + uri + ")")
    # $x$ inline math -> italics (the report uses unicode math otherwise)
    text = re.sub(r"\$([^$]{1,12})\$", r"*\1*", text)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "smarty"], output_format="html5"
    )
    with tempfile.TemporaryDirectory() as td:
        html = Path(td) / "report.html"
        html.write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>modelDNA tech report</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>",
            encoding="utf-8",
        )
        subprocess.run(
            [
                BROWSER,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={PDF_OUT}",
                html.as_uri(),
            ],
            check=True,
            timeout=120,
        )
    print("wrote", PDF_OUT, PDF_OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
