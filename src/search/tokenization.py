from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from typing import Any


TERM_PATTERN = re.compile(r"[가-힣]+|[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)


class KoreanSearchTokenizer:
    """Dependency-free tokenizer combining eojeol-like terms and Hangul n-grams."""

    version = "korean-term-char-ngram-v1"

    def __init__(self, ngram_sizes: tuple[int, ...] = (2, 3)) -> None:
        self.ngram_sizes = ngram_sizes

    def __call__(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens: list[str] = []
        for term in TERM_PATTERN.findall(normalized):
            tokens.append(term)
            if re.fullmatch(r"[가-힣]+", term):
                for size in self.ngram_sizes:
                    if len(term) >= size:
                        tokens.extend(term[index : index + size] for index in range(len(term) - size + 1))
        return tokens


class RegexTokenizer:
    version = "regex-term-v1"

    def __call__(self, text: str) -> list[str]:
        return TERM_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


class WhitespaceTokenizer:
    version = "whitespace-v1"

    def __call__(self, text: str) -> list[str]:
        return unicodedata.normalize("NFKC", text).casefold().split()


def build_tokenizer(config: dict[str, Any] | None = None) -> Callable[[str], list[str]]:
    config = config or {"type": "korean_ngram"}
    tokenizer_type = config.get("type", "korean_ngram")
    if tokenizer_type == "korean_ngram":
        sizes = tuple(int(size) for size in config.get("ngram_sizes", (2, 3)))
        tokenizer = KoreanSearchTokenizer(sizes)
        tokenizer.version = f"korean-term-char-ngram-v1:{','.join(map(str, sizes))}"
        return tokenizer
    if tokenizer_type == "regex":
        return RegexTokenizer()
    if tokenizer_type == "whitespace":
        return WhitespaceTokenizer()
    if tokenizer_type == "morpheme":
        raise ValueError(
            "The morpheme tokenizer is not installed. Configure an analyzer adapter first; "
            "use korean_ngram, regex, or whitespace for the current dependency set."
        )
    raise ValueError(
        f"Unknown BM25 tokenizer {tokenizer_type!r}; choose korean_ngram, regex, whitespace, or morpheme"
    )
