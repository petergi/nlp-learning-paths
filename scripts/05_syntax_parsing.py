"""Syntax parsing with spaCy: POS tagging, NER, and dependency parsing."""


def main() -> None:
    try:
        import spacy
    except ImportError:
        print("spaCy is not installed. Run: pip install spacy")
        return

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found.")
        print("Install it with: python -m spacy download en_core_web_sm")
        return

    # --- POS Tagging ---
    print("=" * 60)
    print("POS TAGGING")
    print("=" * 60)
    sentence: str = "The quick brown fox jumps over the lazy dog."
    doc = nlp(sentence)
    print(f"Sentence: {sentence}\n")
    print(f"{'Token':<12} {'POS':<8} {'Explanation'}")
    print("-" * 50)
    for token in doc:
        print(f"{token.text:<12} {token.pos_:<8} {spacy.explain(token.pos_)}")

    # --- Named Entity Recognition ---
    print(f"\n{'=' * 60}")
    print("NAMED ENTITY RECOGNITION")
    print("=" * 60)
    ner_text: str = (
        "Apple was founded by Steve Jobs in Cupertino, California in 1976."
    )
    doc = nlp(ner_text)
    print(f"Sentence: {ner_text}\n")
    print(f"{'Entity':<20} {'Label':<12} {'Explanation'}")
    print("-" * 55)
    for ent in doc.ents:
        print(f"{ent.text:<20} {ent.label_:<12} {spacy.explain(ent.label_)}")

    # --- Dependency Parsing ---
    print(f"\n{'=' * 60}")
    print("DEPENDENCY PARSING")
    print("=" * 60)
    dep_text: str = "The cat sat on the mat near the window."
    doc = nlp(dep_text)
    print(f"Sentence: {dep_text}\n")
    print(f"{'Token':<12} {'Dep Label':<12} {'Head'}")
    print("-" * 40)
    for token in doc:
        print(f"{token.text:<12} {token.dep_:<12} {token.head.text}")


if __name__ == "__main__":
    main()
