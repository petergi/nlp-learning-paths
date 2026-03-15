"""Simple retrieval-based chatbot using TF-IDF and cosine similarity."""

KNOWLEDGE_BASE: list[dict[str, str]] = [
    {"question": "Hello, how are you?", "answer": "I'm doing great, thanks for asking!"},
    {"question": "What is your name?", "answer": "I'm a simple retrieval-based chatbot."},
    {"question": "What is the weather like today?", "answer": "I don't have live data, but I hope it's sunny where you are!"},
    {"question": "What is natural language processing?", "answer": "NLP is a field of AI focused on the interaction between computers and human language."},
    {"question": "What is machine learning?", "answer": "Machine learning is a subset of AI where systems learn patterns from data without being explicitly programmed."},
    {"question": "What is Python used for?", "answer": "Python is used for web development, data science, machine learning, automation, and much more."},
    {"question": "How do I install Python packages?", "answer": "Use pip: run 'pip install package_name' in your terminal."},
    {"question": "What are transformers in NLP?", "answer": "Transformers are deep learning models that use self-attention mechanisms, powering models like BERT and GPT."},
    {"question": "Tell me a fun fact", "answer": "The word 'set' has the most definitions of any English word -- over 430!"},
    {"question": "Goodbye", "answer": "Goodbye! Have a great day!"},
]

CONFIDENCE_THRESHOLD: float = 0.15


def find_best_match(user_input: str, questions: list[str]) -> tuple[int, float]:
    """Return the index and similarity score of the best matching question."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Missing dependency. Install with: pip install scikit-learn")
        raise SystemExit(1)

    corpus = questions + [user_input]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    best_idx: int = int(similarities.argmax())
    best_score: float = float(similarities[best_idx])
    return best_idx, best_score


def main() -> None:
    questions = [entry["question"] for entry in KNOWLEDGE_BASE]

    print("=== Retrieval-Based Chatbot ===")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        user_input: str = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye!")
            break

        idx, score = find_best_match(user_input, questions)

        if score < CONFIDENCE_THRESHOLD:
            print(f"Bot: I'm not sure about that. (confidence: {score:.2f})\n")
        else:
            matched_q = KNOWLEDGE_BASE[idx]["question"]
            answer = KNOWLEDGE_BASE[idx]["answer"]
            print(f"Bot: {answer}")
            print(f"     [matched: \"{matched_q}\" | confidence: {score:.2f}]\n")


if __name__ == "__main__":
    main()
