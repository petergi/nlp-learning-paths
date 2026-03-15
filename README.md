# NLP with Python Learning Workspace

A structured learning workspace for Natural Language Processing (NLP) in Python, built for people who learn best by following a clear, progressive path through the material.

The concepts and techniques covered here are drawn from various NLP textbooks and resources — what's original is the organization: everything has been distilled into two complementary learning tracks that build on each other chapter by chapter. A scripts track for quick, hands-on experimentation, and a notebooks track for deeper exploration with full explanations and exercises.

## Project Structure

```text
├── .vscode/              # VS Code workspace settings and extensions
├── scripts/              # Track 1: runnable scripts (one per chapter)
├── notebooks/            # Track 2: deep-dive Jupyter notebooks
├── src/nlp_learning/     # shared NLP utility modules
├── tests/                # unit tests for utility modules
├── data/                 # reusable datasets (sentiment, topics, chatbot KB)
└── pyproject.toml
```

## Getting Started

**Prerequisites:** Python 3.10 or later. Verify with:

```bash
python3 --version
```

### Option A: Command Line

1. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -e '.[dev]'
   ```

   For script 09 (machine translation) and notebook deep-learning sections, also install:

   ```bash
   pip install -e '.[dev,transformers]'
   ```

3. **Download NLP models and data:**

   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon omw-1.4
   python -m spacy download en_core_web_sm
   ```

4. **Verify the setup:**

   ```bash
   pytest
   python scripts/01_text_preprocessing.py
   ```

5. **Launch notebooks:**

   ```bash
   jupyter notebook
   ```

### Option B: VS Code

The repository includes workspace settings (`.vscode/`) that configure the Python interpreter and recommended extensions automatically.

1. **Open the project folder** in VS Code (`File > Open Folder...`).

2. **Install recommended extensions** when prompted — or manually install:
   - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
   - [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

3. **Create the virtual environment.** Open the integrated terminal (<kbd>Ctrl</kbd>+<kbd>`</kbd>) and run:

   ```bash
   python3 -m venv .venv
   ```

   VS Code will detect the new environment and ask to select it as the interpreter — click **Yes**. If it doesn't, open the Command Palette (<kbd>Cmd</kbd>/<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>) and run **Python: Select Interpreter**, then choose `.venv`.

4. **Install dependencies** from the terminal (the venv activates automatically):

   ```bash
   pip install -e '.[dev]'
   ```

5. **Download NLP models and data:**

   ```bash
   python -m nltk.downloader punkt punkt_tab stopwords wordnet vader_lexicon omw-1.4
   python -m spacy download en_core_web_sm
   ```

6. **Open any notebook** from the `notebooks/` folder. VS Code will use the `.venv` kernel automatically. If prompted to select a kernel, choose **Python Environments > .venv**.

## Track 1: Scripts

Concise, runnable Python scripts — one per chapter. Great for quick experimentation from the command line.

```bash
python scripts/01_text_preprocessing.py
python scripts/02_text_cleaning.py
# ... through 10_chatbot.py
```

| # | Script | Topics |
| --- | ------ | ------ |
| 01 | `01_text_preprocessing.py` | Tokenization, stopword removal, text cleaning |
| 02 | `02_text_cleaning.py` | Stemming, lemmatization, regex patterns |
| 03 | `03_feature_engineering.py` | Bag-of-Words, TF-IDF, top terms |
| 04 | `04_language_modeling.py` | Bigram model, probabilities, text generation |
| 05 | `05_syntax_parsing.py` | POS tagging, NER, dependency parsing (spaCy) |
| 06 | `06_sentiment_analysis.py` | VADER, TextBlob, sklearn classifier comparison |
| 07 | `07_topic_modeling.py` | LSA (sklearn), LDA (gensim) |
| 08 | `08_text_summarization.py` | Extractive summarization with NLTK |
| 09 | `09_machine_translation.py` | English-to-French with Hugging Face Transformers |
| 10 | `10_chatbot.py` | Retrieval-based chatbot with TF-IDF matching |

## Track 2: Notebooks

Full chapter-by-chapter Jupyter notebooks with explanations, code, and exercises with solutions.

```bash
jupyter notebook
```

| # | Notebook | Cells | Key Topics |
| --- | -------- | ----- | ---------- |
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

## Shared Utilities (`src/nlp_learning/`)

Reusable functions used by both scripts and notebooks:

- **preprocessing** — `clean_text`, `tokenize`, `remove_stopwords`
- **text_cleaning** — `stem_words`, `lemmatize_words`, `extract_patterns`
- **features** — `build_bow_matrix`, `build_tfidf_matrix`
- **similarity** — `find_best_match` (TF-IDF + cosine similarity)
