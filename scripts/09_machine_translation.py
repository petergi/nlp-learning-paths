"""Machine translation from English to French using Hugging Face transformers."""


def main() -> None:
    try:
        from transformers import pipeline
    except ImportError:
        print("Missing dependencies. Install with:")
        print("  pip install transformers torch sentencepiece")
        return

    print("=== Machine Translation (EN -> FR) ===\n")
    print("Loading model: Helsinki-NLP/opus-mt-en-fr ...")

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

    sentences: list[str] = [
        "Natural language processing is a fascinating field of computer science.",
        "The weather in Paris is beautiful during the spring.",
        "Machine learning models require large amounts of training data.",
        "I would like to order a coffee and a croissant, please.",
    ]

    print()
    for i, sentence in enumerate(sentences, 1):
        result = translator(sentence, max_length=128)
        translation: str = result[0]["translation_text"]
        print(f"  [{i}] EN: {sentence}")
        print(f"      FR: {translation}\n")

    print("Translation complete.")


if __name__ == "__main__":
    main()
