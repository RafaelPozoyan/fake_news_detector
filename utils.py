import re
from typing import Dict

_URL_RE = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+", re.IGNORECASE)
_TAG_RE = re.compile(r"<.*?>")
_NON_CYRILLIC_RE = re.compile(r"[^а-яё\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _TAG_RE.sub(" ", t)
    t = _NON_CYRILLIC_RE.sub(" ", t)
    t = _MULTISPACE_RE.sub(" ", t).strip()
    return t


def combine_features(headline_clean: str, body_clean: str, max_body_words: int = 100) -> str:
    body_words = body_clean.split()[:max_body_words]
    return f"{headline_clean} {' '.join(body_words)}".strip()


LABEL_MAPPING: Dict[int, str] = {0: 'fake', 1: 'real'}


