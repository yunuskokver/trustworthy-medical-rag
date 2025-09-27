import re, unicodedata

def normalize_text(text: str) -> str:
    """Lowercase, remove HTML, strip non-ASCII, standardize punctuation, trim whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("utf-8","ignore")
    text = text.replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"').replace("–","-").replace("—","-")
    text = re.sub(r"\s+"," ", text).strip()
    return text
