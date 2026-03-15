"""Extractive text summarization using NLTK word frequency scoring."""

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def _ensure_nltk_data() -> None:
    """Download required NLTK resources if not already present."""
    import nltk
    for resource in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)


def score_sentences(text: str) -> dict[str, float]:
    """Score each sentence by normalized word frequency (excluding stopwords)."""
    stop_words = set(stopwords.words("english"))
    words = [
        w.lower() for w in word_tokenize(text)
        if w.isalnum() and w.lower() not in stop_words
    ]

    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    max_freq = max(freq.values()) if freq else 1

    sentences = sent_tokenize(text)
    scores: dict[str, float] = {}
    for sent in sentences:
        sent_words = [w.lower() for w in word_tokenize(sent) if w.isalnum()]
        scores[sent] = sum(freq.get(w, 0) / max_freq for w in sent_words)
    return scores


def summarize(text: str, num_sentences: int = 2) -> str:
    """Return the top N highest-scoring sentences in original order."""
    scores = score_sentences(text)
    ranked = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    original_order = [s for s in scores if s in ranked]
    return " ".join(original_order)


def main() -> None:
    _ensure_nltk_data()

    text = (
        "Natural language processing enables computers to understand human language. "
        "It combines computational linguistics with machine learning and deep learning models. "
        "Applications include machine translation, sentiment analysis, and text summarization. "
        "NLP has become essential in modern search engines and virtual assistants. "
        "Recent advances in transformer models have significantly improved NLP performance."
    )

    print("=== Extractive Text Summarization ===\n")
    print(f"Original ({len(sent_tokenize(text))} sentences):\n{text}\n")

    scores = score_sentences(text)
    print("Sentence scores:")
    for sent, score in scores.items():
        print(f"  [{score:.2f}] {sent}")

    summary = summarize(text, num_sentences=2)
    print(f"\nSummary (top 2 sentences):\n{summary}")


if __name__ == "__main__":
    main()
