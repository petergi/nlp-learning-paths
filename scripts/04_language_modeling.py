import random
from collections import Counter


def main() -> None:
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat chased the dog",
        "the dog chased the cat around the house",
        "the mat was on the floor of the house",
    ]

    bigrams: list[tuple[str, str]] = []
    for sentence in corpus:
        words = sentence.split()
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))

    bigram_counts = Counter(bigrams)
    word_counts = Counter(w for w, _ in bigrams)

    print("=== Bigram Language Model ===")
    print(f"Total bigrams: {len(bigrams)}")
    print(f"Unique bigrams: {len(bigram_counts)}")

    print("\n=== Most Common Bigrams ===")
    for bigram, count in bigram_counts.most_common(8):
        prob = count / word_counts[bigram[0]]
        print(f"  {bigram[0]:>6} -> {bigram[1]:<8}  count={count}  P={prob:.2f}")

    print("\n=== Bigram Probabilities for 'the' ===")
    the_bigrams = {b: c for b, c in bigram_counts.items() if b[0] == "the"}
    for (_, next_word), count in sorted(the_bigrams.items(), key=lambda x: -x[1]):
        prob = count / word_counts["the"]
        print(f"  P({next_word} | the) = {prob:.2f}")

    print("\n=== Generated Sentence ===")
    random.seed(42)
    word = "the"
    sentence = [word]
    for _ in range(8):
        candidates = [(b[1], c) for b, c in bigram_counts.items() if b[0] == word]
        if not candidates:
            break
        words, weights = zip(*candidates)
        word = random.choices(words, weights=weights, k=1)[0]
        sentence.append(word)
    print(" ".join(sentence))


if __name__ == "__main__":
    main()
