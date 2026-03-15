# NLP with Python Learning Workspace

A practical, beginner-friendly workspace to learn Natural Language Processing (NLP) in Python by building small, focused scripts.

## What You Will Learn

1. Text normalization and tokenization
2. Stopword removal and lightweight preprocessing
3. Sentiment analysis with bag-of-words features
4. Basic text classification with scikit-learn

## Project Structure

- `src/nlp_learning/`: reusable NLP utility code
- `scripts/`: step-by-step runnable examples
- `data/`: sample datasets for exercises
- `tests/`: basic tests to validate preprocessing logic

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:

```bash
pip install -e .[dev]
```

1. Download required NLTK resources (one-time):

```bash
python -m nltk.downloader punkt stopwords
```

1. Run scripts in order:

```bash
python scripts/01_text_preprocessing.py
python scripts/02_sentiment_analysis.py
python scripts/03_text_classification.py
```

1. Run tests:

```bash
pytest
```

## Study Roadmap

1. **Week 1**: text cleanup, tokenization, stopwords, vocabulary basics
2. **Week 2**: bag-of-words and TF-IDF features
3. **Week 3**: supervised models for text classification
4. **Week 4**: extend with spaCy pipelines (NER, POS tagging)

## Interactive Notebook Track

Open Jupyter from the project root:

```bash
jupyter notebook
```

Then run these notebooks in order:

1. `notebooks/01_week1_text_basics.ipynb`
2. `notebooks/02_week2_features_tfidf.ipynb`
3. `notebooks/03_week3_classification.ipynb`
4. `notebooks/04_week4_spacy_intro.ipynb`

spaCy model note for Week 4:

```bash
python -m spacy download en_core_web_sm
```

## NLP with Python Book — Notebook Track

Full chapter-by-chapter notebooks based on the NLP with Python textbook. Each includes runnable code, explanations, and exercises with solutions.

| # | Notebook | Cells | Key Topics |
| --- | ---------- | ------- | ------------ |
| 01 | [Introduction to NLP](notebooks/01_Introduction_to_NLP.ipynb) | 46 | Tokenization, NLP pipelines, NLTK/spaCy/TextBlob |
| 02 | [Basic Text Processing](notebooks/02_Basic_Text_Processing.ipynb) | 64 | Stop words, stemming, lemmatization, regex, tokenization |
| 03 | [Feature Engineering](notebooks/03_Feature_Engineering_for_NLP.ipynb) | 64 | BoW, TF-IDF, Word2Vec, GloVe, BERT embeddings |
| 04 | [Language Modeling](notebooks/04_Language_Modeling.ipynb) | 53 | N-grams, HMMs, RNNs, LSTMs |
| 05 | [Syntax and Parsing](notebooks/05_Syntax_and_Parsing.ipynb) | 51 | POS tagging, NER, dependency parsing |
| 06 | [Sentiment Analysis](notebooks/06_Sentiment_Analysis.ipynb) | 57 | Rule-based, ML (LogReg/NB/SVM), deep learning (CNN/LSTM/BERT) |
| 07 | [Topic Modeling](notebooks/07_Topic_Modeling.ipynb) | 43 | LSA, LDA, HDP, coherence evaluation |
| 08 | [Text Summarization](notebooks/08_Text_Summarization.ipynb) | 40 | Extractive (NLTK/TextRank), abstractive (BART/T5), ROUGE |
| 09 | [Machine Translation](notebooks/09_Machine_Translation.ipynb) | 42 | Seq2Seq, attention mechanisms, Transformers (T5) |
| 10 | [Introduction to Chatbots](notebooks/10_Introduction_to_Chatbots.ipynb) | 44 | Rule-based, retrieval-based, generative, hybrid |
| 11 | [Chatbot Project](notebooks/11_Personal_Assistant_Chatbot_Project.ipynb) | 51 | Full personal assistant with NLP engine, APIs, deployment |
| 12 | [News Aggregator Project](notebooks/12_News_Aggregator_Project.ipynb) | 56 | News collection, summarization, topic modeling, Flask UI |
| 13 | [Sentiment Dashboard Project](notebooks/13_Sentiment_Analysis_Dashboard_Project.ipynb) | 58 | SMOTE, LogReg/LSTM, Flask+Plotly dashboard, Heroku |

## Next Exercises

- Replace CountVectorizer with TF-IDF in `03_text_classification.py`
- Add your own dataset to `data/`
- Compare Logistic Regression vs Naive Bayes for classification
