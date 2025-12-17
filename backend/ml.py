import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ModelConfig:
    max_features: int = 50_000
    ngram_range: tuple = (1, 2)
    C: float = 2.0
    max_iter: int = 1000


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_USER_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = _URL_RE.sub(" URL ", text)
    text = _USER_RE.sub(" USER ", text)
    text = text.replace("#", " #")
    text = _WS_RE.sub(" ", text)
    return text


def build_pipeline(cfg: ModelConfig | None = None) -> Pipeline:
    cfg = cfg or ModelConfig()

    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        lowercase=True,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=2,
    )

    clf = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        n_jobs=1,
        class_weight="balanced",
        multi_class="auto",
    )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])
