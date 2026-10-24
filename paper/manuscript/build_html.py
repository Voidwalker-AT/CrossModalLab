"""Printable portable article. Authoring uses existing Python Markdown only."""
from pathlib import Path


def main():
    import markdown
    import re
    from html import escape, unescape
    directory = Path(__file__).resolve().parent
    body = markdown.markdown((directory / "draft.md").read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code"])
    body = re.sub(r'<p>(<img[^>]+>)</p>', r'<figure>\1</figure>', body)
    style = """
    @page {size:A4;margin:18mm}
    *{box-sizing:border-box}body{margin:0;color:#171d20;background:#f1f2ee;font:11pt/1.45 Georgia,'Times New Roman',serif}
    main{max-width:1040px;margin:24px auto;background:white;padding:38px;columns:2;column-gap:32px}
    h1{column-span:all;text-align:center;font:700 20pt/1.2 Georgia,serif;margin:0 0 20px}
    h2{font-size:12pt;margin:17px 0 8px;break-after:avoid}p{margin:0 0 10px;text-align:justify}
    code{font:8pt/1.4 Consolas,monospace;overflow-wrap:anywhere}a{color:#12574b;overflow-wrap:anywhere}
    table{border-collapse:collapse;width:100%;font-size:8pt;break-inside:avoid;margin:10px 0}
    td,th{padding:4px;border-bottom:1px solid #bbc4bd;text-align:left;overflow-wrap:anywhere}
    figure{column-span:all;break-inside:avoid;margin:18px 0}figure img{width:100%;height:auto;display:block}
    figcaption{font-size:9pt;margin-top:7px}pre{white-space:pre-wrap}img{max-width:100%}
    .wide-table{column-span:all;overflow-x:auto;margin:18px 0;break-inside:avoid}
    .wide-table table{min-width:820px;font-size:9pt;margin:0}
    .wide-table td{white-space:nowrap;overflow-wrap:normal}
    .wide-table td:first-child{white-space:normal;min-width:185px;max-width:220px;word-break:break-word}
    .wide-table th{font-size:9pt}
    @media(max-width:760px){main{columns:1;padding:20px;margin:0}body{font-size:11pt}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0}a{color:inherit}}
    """
    body = re.sub(r'<figure>(<img[^>]*alt="([^"]*)"[^>]*>)</figure>',
                  lambda m: '<figure>'+m[1]+'<figcaption>'+escape(unescape(m[2]))+'</figcaption></figure>', body)
    body = re.sub(r'<table>.*?</table>',
                  lambda m: '<div class="wide-table">'+m[0]+'</div>' if 'New development case' in m[0] else m[0],
                  body, flags=re.DOTALL)
    out = directory / "draft.html"
    out.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">'
                   '<title>CrossModalLab — preliminary university paper</title><style>'+style+
                   '</style></head><body><main>'+body+'</main></body></html>\n', encoding="utf-8")
    print("HTML_ARTICLE_WRITTEN", out)


if __name__ == "__main__":
    main()
