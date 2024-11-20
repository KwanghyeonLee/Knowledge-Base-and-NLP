from string import punctuation
from typing import Dict, List, Tuple

from nltk.corpus import stopwords

from src.data2tensor import lemmatizer

STOPWORDS = set(stopwords.words("english"))


def lower_strip_split(sentence: str) -> List[str]:
    """
    @param sentence: raw string sentence.
    @return: lowered, stripped sentence split by single space.

    """
    return sentence.lower().strip().split()


# Remove stopwords
def remove_stopwords(terms: List[str]) -> List[str]:
    """
    @param terms: terms already lowered, striped, split
    @return: list of terms without stopwords
    """
    return list(filter(lambda x: x not in STOPWORDS, terms))


# Lemmatization
def lemmatize_sentence(terms: List[str]) -> List[str]:
    """
    @param terms: terms already lowered, striped, split
    @return: lemmatized terms
    """
    lem_terms = [lemmatizer.lemmatize(term) for term in terms]
    return [lem_term for lem_term in lem_terms if lem_term]


# Remove punctuation at the start/end of each word in a sentence
def remove_punctuation(terms: List[str]) -> List[str]:
    """
    @param terms: terms already lowered, striped, split
    @return:
    """
    punc_removed = [word.strip(punctuation) for word in terms]
    return [word for word in punc_removed if word]


# Regularize string: remove punctuation -> remove stopwords
def regularize_string(sentence: str) -> str:
    """
    @param sentence: raw sentence
    @return: regularized sentence
    """
    return " ".join(remove_stopwords(remove_punctuation(lower_strip_split(sentence))))
