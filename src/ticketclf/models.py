"""Ticket classifiers: a keyword-rule baseline and two TF-IDF + logistic regression models."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

CATEGORIES = ["card_issue", "fraud_dispute", "login_access", "transfer_payment", "account_fees",
              "loan_inquiry", "technical_bug", "account_closure"]


class KeywordRules:
    """The kind of routing rules many help desks start with. Checked in priority order."""

    name = "keyword_rules"
    RULES = [
        ("fraud_dispute", r"fraud|unauthori[sz]ed|didn.?t make|not mine|aren.?t mine|stolen|dispute|scam|hacked|charged twice"),
        ("account_closure", r"close (my|the|a|all)|closing|cancel my credit card"),
        ("login_access", r"log ?in|password|locked|sign in|verification code|two.factor|face id|username|logged out"),
        ("loan_inquiry", r"loan|mortgage|refinanc|payoff|credit score|home equity"),
        ("account_fees", r"\bfee|waive|interest"),
        ("transfer_payment", r"transfer|wire|zelle|ach|deposit|bill pay|payment|paycheck|sent"),
        ("technical_bug", r"app|crash|error|website|load|notification|blank screen|slow|pdf"),
        ("card_issue", r"card|pin|atm|declin|contactless"),
    ]

    def fit(self, X, y):
        self.classes_ = np.array(CATEGORIES)
        self._compiled = [(c, re.compile(p, re.I)) for c, p in self.RULES]
        return self

    def predict(self, X):
        out = []
        for t in X:
            for c, rx in self._compiled:
                if rx.search(t):
                    out.append(c)
                    break
            else:
                out.append("card_issue")  # most common queue as the fallback
        return np.array(out)

    def predict_proba(self, X):
        pred = self.predict(X)
        return (pred[:, None] == self.classes_[None, :]).astype(float)


def tfidf_words(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(C=8, max_iter=3000, class_weight="balanced", random_state=seed)),
    ])


def tfidf_words_chars(seed: int = 42) -> Pipeline:
    """Adds character n-grams, which hold up against typos and unseen word forms."""
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                      sublinear_tf=True, max_features=60_000)),
        ])),
        ("clf", LogisticRegression(C=8, max_iter=3000, class_weight="balanced", random_state=seed)),
    ])
