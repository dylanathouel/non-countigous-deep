import markdown
import subprocess
import sys
import os
from pathlib import Path

md_path = Path(sys.argv[1])
html_path = md_path.with_suffix(".html")
pdf_path = md_path.with_suffix(".pdf")

md_text = md_path.read_text(encoding="utf-8")
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

css = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: #222; font-size: 11pt; line-height: 1.45; }
h1 { font-size: 20pt; color: #1a3d6d; border-bottom: 2px solid #1a3d6d; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 15pt; color: #1a3d6d; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 22px; }
h3 { font-size: 12pt; color: #2a5a9c; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 6px 9px; text-align: left; }
th { background: #e8eef7; color: #1a3d6d; font-weight: 600; }
tr:nth-child(even) td { background: #f7f9fc; }
strong { color: #1a3d6d; }
blockquote { border-left: 4px solid #2a5a9c; background: #f0f4fa; margin: 10px 0; padding: 8px 14px; color: #333; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 10pt; }
em { color: #666; }
"""

html_doc = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>{md_path.stem}</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_path.write_text(html_doc, encoding="utf-8")

chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([
    chrome,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    f"file://{html_path.absolute()}",
], check=True)

print(f"PDF: {pdf_path}")
