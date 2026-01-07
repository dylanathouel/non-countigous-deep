import markdown
import os

# Read the markdown content
md_path = "/Users/macdedylan/.gemini/antigravity/brain/7aa9ef04-da2d-4288-b33c-5adf411078e8/walkthrough.md"
with open(md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert to HTML
html_content = markdown.markdown(text, extensions=['tables'])

# Add some CSS for a professional look (approximating a PDF report style)
css = """
<style>
    body { font-family: 'Helvetica', 'Arial', sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }
    h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
    th { background-color: #f2f2f2; color: #2c3e50; font-weight: bold; }
    tr:nth-child(even) { background-color: #f9f9f9; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 0 10px rgba(0,0,0,0.1); margin: 20px 0; }
    code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: 'Courier New', monospace; }
</style>
"""

final_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Rapport Final Experimentation 3D</title>
    {css}
</head>
<body>
    {html_content}
</body>
</html>
"""

# Write to file in the desktop for easy access
output_path = "/Users/macdedylan/Desktop/Rapport_Experimentation_3D.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

print(f"Report generated at: {output_path}")
