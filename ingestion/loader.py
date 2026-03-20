import fitz

def load_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": i})
    doc.close()
    return pages