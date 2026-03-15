import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main() -> None:
    corpus = [
        "Natural language processing is a field of AI.",
        "Machine learning powers many NLP applications.",
        "Text classification is an important NLP task.",
        "Deep learning models improve language understanding.",
        "Sentiment analysis detects opinions in text data.",
    ]

    print("=== Bag of Words (CountVectorizer) ===")
    bow = CountVectorizer()
    bow_matrix = bow.fit_transform(corpus)
    print("Vocabulary size:", len(bow.get_feature_names_out()))
    print("Matrix shape:", bow_matrix.shape)
    print("Sample features:", bow.get_feature_names_out()[:10].tolist())

    print("\n=== TF-IDF (TfidfVectorizer) ===")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus)
    print("Matrix shape:", tfidf_matrix.shape)

    print("\n=== Top TF-IDF Terms per Document ===")
    feature_names = tfidf.get_feature_names_out()
    for i, doc in enumerate(corpus):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = np.argsort(scores)[-3:][::-1]
        top_terms = [(feature_names[j], round(scores[j], 3)) for j in top_indices]
        print(f"Doc {i + 1}: {top_terms}")


if __name__ == "__main__":
    main()
