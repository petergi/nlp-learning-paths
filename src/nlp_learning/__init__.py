"""Utilities for the NLP Python learning workspace."""

from nlp_learning.features import build_bow_matrix, build_tfidf_matrix
from nlp_learning.preprocessing import clean_text, remove_stopwords, tokenize
from nlp_learning.similarity import find_best_match
from nlp_learning.text_cleaning import extract_patterns, lemmatize_words, stem_words

__all__ = [
    "clean_text",
    "tokenize",
    "remove_stopwords",
    "stem_words",
    "lemmatize_words",
    "extract_patterns",
    "build_bow_matrix",
    "build_tfidf_matrix",
    "find_best_match",
]
