import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from ticketclf.llm import parse  # noqa: E402
from ticketclf.models import CATEGORIES, KeywordRules  # noqa: E402
from ticketclf.pipeline import split  # noqa: E402
from ticketclf.routing import pick_threshold, routing_curve  # noqa: E402


@pytest.fixture(scope="module")
def df():
    d = generate(n=3000, seed=11)
    d["category_true"] = d["template_id"].str.split("#").str[0]
    return d


def test_all_categories_present(df):
    assert set(df["category"]) == set(CATEGORIES)


def test_unseen_split_has_no_template_leakage(df):
    train, seen, unseen, held = split(df)
    assert not train["template_id"].isin(held).any()
    assert not train["secondary_template_id"].isin(held).any()
    assert unseen["template_id"].isin(held).all()


def test_keyword_rules_basic():
    m = KeywordRules().fit(None, None)
    assert m.predict(["I want to close my account"])[0] == "account_closure"
    assert m.predict(["there is a charge I didn't make"])[0] == "fraud_dispute"


def test_routing_curve_monotone_automation():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=500)
    classes = np.array(list("abcd"))
    y = classes[rng.integers(0, 4, 500)]
    c = routing_curve(y, proba, classes)
    assert (np.diff(c["automation_rate"]) <= 1e-12).all()


def test_pick_threshold_meets_target():
    curve = pd.DataFrame({"threshold": [0.5, 0.7, 0.9], "automation_rate": [0.9, 0.7, 0.4],
                          "auto_routed_accuracy": [0.9, 0.96, 0.99]})
    assert pick_threshold(curve, 0.95)["threshold"] == 0.7


def test_llm_parse_handles_messy_output():
    assert parse('{"queue": "login_access", "confidence": 0.9}')["queue"] == "login_access"
    assert parse('Sure! {"queue": "fraud_dispute", "confidence": "0.8"}')["confidence"] == 0.8
    assert parse("The answer is technical_bug")["queue"] == "technical_bug"
    assert parse("no idea")["queue"] == "unparseable"
