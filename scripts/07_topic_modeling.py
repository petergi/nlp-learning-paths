"""Topic modeling with LSA (sklearn) and LDA (gensim)."""

from __future__ import annotations

CORPUS: list[str] = [
    "Machine learning algorithms process large datasets efficiently.",
    "Neural networks power modern artificial intelligence systems.",
    "Python is a popular programming language for data science.",
    "Chop the onions and garlic before heating the olive oil.",
    "Bake the cake at 350 degrees for thirty minutes.",
    "Season the chicken with salt, pepper, and fresh herbs.",
    "The quarterback threw a perfect pass for the touchdown.",
    "The team won the championship game in overtime.",
    "Soccer players train daily to improve speed and endurance.",
    "Cloud computing enables scalable software deployment worldwide.",
]


def run_lsa(corpus: list[str], n_topics: int = 3) -> None:
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("scikit-learn not installed. Run: pip install scikit-learn")
        return

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()

    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    svd.fit(tfidf_matrix)

    print("=" * 60)
    print("LSA (Latent Semantic Analysis) — sklearn TruncatedSVD")
    print("=" * 60)
    for i, component in enumerate(svd.components_):
        top_indices = component.argsort()[-6:][::-1]
        top_words: list[str] = [feature_names[j] for j in top_indices]
        print(f"  Topic {i+1}: {', '.join(top_words)}")


def run_lda(corpus: list[str], n_topics: int = 3) -> None:
    try:
        from gensim import corpora, models
    except ImportError:
        print("gensim not installed. Run: pip install gensim")
        return

    tokenized: list[list[str]] = [doc.lower().split() for doc in corpus]
    # Simple stop word removal
    stop_words: set[str] = {
        "the", "a", "an", "is", "in", "for", "and", "to", "of", "at", "with",
    }
    tokenized = [
        [w for w in doc if w not in stop_words and len(w) > 2]
        for doc in tokenized
    ]

    dictionary = corpora.Dictionary(tokenized)
    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = models.LdaModel(
        bow_corpus,
        num_topics=n_topics,
        id2word=dictionary,
        passes=20,
        random_state=42,
    )

    print(f"\n{'=' * 60}")
    print("LDA (Latent Dirichlet Allocation) — gensim")
    print("=" * 60)
    for i in range(n_topics):
        top_words: list[str] = [
            word for word, _ in lda.show_topic(i, topn=6)
        ]
        print(f"  Topic {i+1}: {', '.join(top_words)}")


def main() -> None:
    print("Corpus:")
    for i, doc in enumerate(CORPUS, 1):
        print(f"  {i:>2}. {doc}")
    print()

    run_lsa(CORPUS)
    run_lda(CORPUS)

    print("\nNote: Topics are unlabeled — inspect top words to infer themes.")
    print("Expected themes: technology, cooking, sports.")


if __name__ == "__main__":
    main()
