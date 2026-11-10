import fitz


def load_pdf(data: bytes) -> list[dict]:
    """Extract the text of every page that has any. Pages are numbered from 1."""
    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": i})
    return pages
