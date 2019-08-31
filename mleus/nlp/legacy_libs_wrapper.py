import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SklearnWrapper(object):

    def __init__(self, corpus):
        self.corpus = corpus

        self.tfidf_vectorizer = TfidfVectorizer()

    def train_tfidf(self, corpus):
        if isinstance(corpus, list):
            self.tfidf_vectorizer.fit(corpus)
        else:
            self.tfidf_vectorizer.fit([corpus])

    def get_tfidf_vectors(self, text):
        if isinstance(text, list):
            return self.tfidf_vectorizer.transform(text).tolist()
        else:
            return self.tfidf_vectorizer.transform(text).tolist()[0]

    def get_important_words(self, text, n_words=20):
        pass

