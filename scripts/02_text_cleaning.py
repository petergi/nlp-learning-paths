import re

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("wordnet", quiet=True)


def main() -> None:
    words = ["running", "better", "geese", "studies", "happily", "wolves"]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    print("=== Stemming vs Lemmatization ===")
    print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
    print("-" * 36)
    for word in words:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        print(f"{word:<12} {stem:<12} {lemma:<12}")

    print("\n=== Regex: Extract Emails ===")
    text = "Contact us at support@example.com or sales@company.org for info."
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    print("Text:", text)
    print("Emails found:", emails)

    print("\n=== Regex: Clean HTML Tags ===")
    html = "<p>Hello <b>world</b>! Visit <a href='#'>here</a>.</p>"
    clean = re.sub(r"<[^>]+>", "", html)
    print("HTML:", html)
    print("Cleaned:", clean)


if __name__ == "__main__":
    main()
