"""Simple retrieval-based chatbot using TF-IDF and cosine similarity."""

import csv
from pathlib import Path

from nlp_learning.similarity import find_best_match

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "chatbot_kb.csv"

CONFIDENCE_THRESHOLD: float = 0.15


def _load_knowledge_base(path: Path) -> list[dict[str, str]]:
    """Load Q&A pairs from a CSV with ``question`` and ``answer`` columns.

    Skips rows with missing or blank ``question``/``answer`` fields.
    """
    entries: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"question", "answer"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: missing required 'question'/'answer' columns")
        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                entries.append({"question": q, "answer": a})
    if not entries:
        raise ValueError(f"{path}: no valid Q&A pairs found")
    return entries


def main() -> None:
    try:
        kb = _load_knowledge_base(DATA_PATH)
    except OSError:
        print(f"Knowledge base not found: {DATA_PATH}")
        print("Make sure you're running from the repo root.")
        return
    except ValueError as exc:
        print(f"Bad knowledge base: {exc}")
        return

    questions = [entry["question"] for entry in kb]

    print("=== Retrieval-Based Chatbot ===")
    print("Type your question (or 'quit' to exit).\n")

    while True:
        user_input: str = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye!")
            break

        try:
            idx, score = find_best_match(user_input, questions)
        except RuntimeError:
            print("Missing dependency: pip install scikit-learn")
            return

        if score < CONFIDENCE_THRESHOLD:
            print(
                "Bot: I'm not sure about that. "
                f"(confidence: {score:.2f})\n"
            )
        else:
            matched_q = kb[idx]["question"]
            answer = kb[idx]["answer"]
            print(f"Bot: {answer}")
            print(
                f"     [matched: \"{matched_q}\" "
                f"| confidence: {score:.2f}]\n"
            )


if __name__ == "__main__":
    main()
