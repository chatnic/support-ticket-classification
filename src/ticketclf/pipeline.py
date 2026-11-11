"""Train and evaluate ticket classifiers, including on phrasings never seen in training.

    PYTHONPATH=src python -m ticketclf.pipeline --data data/raw/tickets.csv --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ticketclf.models import CATEGORIES, KeywordRules, tfidf_words, tfidf_words_chars  # noqa: E402
from ticketclf.routing import pick_threshold, routing_curve  # noqa: E402

SEED = 42
HELDOUT_TEMPLATES_PER_CLASS = 3
TARGET_ACCURACY = 0.97


def split(df: pd.DataFrame):
    """Three-way split: train, a random test set, and an 'unseen phrasing' test set built from
    whole templates that never appear in training (a much harder, more honest test)."""
    rng = np.random.default_rng(SEED)
    held = []
    for cat in CATEGORIES:
        tids = sorted(df.loc[df.category_true == cat, "template_id"].unique())
        held += list(rng.choice(tids, size=HELDOUT_TEMPLATES_PER_CLASS, replace=False))
    unseen = df[df.template_id.isin(held)]
    # Held-out phrasings must not leak into training as a ticket's secondary topic either.
    rest = df[~df.template_id.isin(held) & ~df.secondary_template_id.isin(held)]
    train, test_seen = train_test_split(rest, test_size=0.2, stratify=rest["category"], random_state=SEED)
    return train, test_seen, unseen, held


def run(data: Path, out: Path) -> dict:
    df = pd.read_csv(data)
    # template_id encodes the intended category; keep it for the split only.
    df["category_true"] = df["template_id"].str.split("#").str[0]
    train, test_seen, test_unseen, held = split(df)
    print(f"train {len(train):,}  test (seen phrasings) {len(test_seen):,}  test (unseen phrasings) {len(test_unseen):,}")

    models = {"keyword_rules": KeywordRules(), "tfidf_words": tfidf_words(), "tfidf_words_chars": tfidf_words_chars()}
    results, probas = {}, {}
    for name, m in models.items():
        m.fit(train["text"].values, train["category"].values)
        res = {}
        for split_name, t in (("seen", test_seen), ("unseen", test_unseen)):
            pred = m.predict(t["text"].values)
            res[split_name] = {"accuracy": accuracy_score(t["category"], pred),
                               "macro_f1": f1_score(t["category"], pred, average="macro")}
        results[name] = res
        probas[name] = m.predict_proba(test_unseen["text"].values)
        print(f"  {name:<18} seen acc {res['seen']['accuracy']:.3f} F1 {res['seen']['macro_f1']:.3f} | "
              f"unseen acc {res['unseen']['accuracy']:.3f} F1 {res['unseen']['macro_f1']:.3f}")

    best = "tfidf_words_chars"
    model = models[best]
    classes = model.classes_
    y_seen, y_unseen = test_seen["category"].values, test_unseen["category"].values
    p_seen = model.predict_proba(test_seen["text"].values)
    pred_seen = classes[p_seen.argmax(1)]
    pred_unseen = model.predict(test_unseen["text"].values)

    # Pick the confidence threshold on familiar traffic, then see what it does on new phrasings.
    curve_seen = routing_curve(y_seen, p_seen, classes)
    curve_unseen = routing_curve(y_unseen, probas[best], classes)
    op = pick_threshold(curve_seen, TARGET_ACCURACY)
    op_unseen = curve_unseen.loc[np.isclose(curve_unseen.threshold, op.threshold)].iloc[0]

    def urgent_recall(y, pred):
        urgent = np.isin(y, ["fraud_dispute", "login_access"])
        return float(np.isin(pred[urgent], ["fraud_dispute", "login_access"]).mean())

    report = pd.DataFrame(classification_report(y_seen, pred_seen, output_dict=True)).T
    cm = pd.DataFrame(confusion_matrix(y_seen, pred_seen, labels=CATEGORIES), index=CATEGORIES, columns=CATEGORIES)

    metrics = {
        "split_sizes": {"train": len(train), "test_seen": len(test_seen), "test_unseen": len(test_unseen)},
        "heldout_templates": held,
        "results": results,
        "selected_model": best,
        "routing_threshold": float(op.threshold),
        "routing_seen": {"automation_rate": float(op.automation_rate),
                         "auto_routed_accuracy": float(op.auto_routed_accuracy)},
        "routing_unseen_same_threshold": {"automation_rate": float(op_unseen.automation_rate),
                                          "auto_routed_accuracy": float(op_unseen.auto_routed_accuracy)},
        "urgent_ticket_recall": {"seen": urgent_recall(y_seen, pred_seen),
                                 "unseen": urgent_recall(y_unseen, pred_unseen)},
        "seen_accuracy_vs_intended_label": float((pred_seen == test_seen["category_true"].values).mean()),
    }
    curve = curve_seen.merge(curve_unseen, on="threshold", suffixes=("_seen", "_unseen"))

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    report.round(3).to_csv(out / "tables" / "per_class_seen.csv")
    cm.to_csv(out / "tables" / "confusion_matrix_seen.csv")
    curve.round(4).to_csv(out / "tables" / "routing_curve.csv", index=False)
    pd.DataFrame([{"model": k, "split": s, **v} for k, r in results.items() for s, v in r.items()]) \
        .round(4).to_csv(out / "tables" / "model_comparison.csv", index=False)
    errors = test_unseen.assign(predicted=pred_unseen, confidence=probas[best].max(1))
    errors[(errors.predicted != errors.category) & (errors.predicted != errors.category_true)].sort_values("confidence", ascending=False) \
        .head(40)[["ticket_id", "text", "category", "predicted", "confidence"]] \
        .to_csv(out / "tables" / "top_errors_unseen.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/ticket_classifier.joblib")
    figures(results, cm, curve, op, op_unseen, out / "figures")
    print(json.dumps({k: v for k, v in metrics.items() if k != "heldout_templates"}, indent=2, default=float))
    return metrics


def figures(results, cm, curve, op, op_unseen, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    labels = {"keyword_rules": "Keyword rules", "tfidf_words": "TF-IDF words + LR",
              "tfidf_words_chars": "TF-IDF words + chars + LR"}
    colors = ["#9aa5b1", "#6c8ebf", "#1f4e8c"]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(2)
    for i, (k, c) in enumerate(zip(results, colors)):
        ax.bar(x + (i - 1) * 0.26, [results[k]["seen"]["macro_f1"], results[k]["unseen"]["macro_f1"]], 0.26,
               label=labels[k], color=c)
    ax.set_xticks(x, ["Familiar phrasings", "Unseen phrasings"])
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Classifier quality: familiar vs. never-seen wording")
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout(); fig.savefig(fig_dir / "model_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(curve.automation_rate_seen, curve.auto_routed_accuracy_seen, lw=2, color="#1f4e8c",
            label="familiar phrasings")
    ax.plot(curve.automation_rate_unseen, curve.auto_routed_accuracy_unseen, lw=2, color="#e08a2c",
            label="never-seen phrasings")
    for pt, lab, off in ((op, "familiar", (-150, -75)), (op_unseen, "never-seen", (45, -40))):
        ax.scatter([pt.automation_rate], [pt.auto_routed_accuracy], color="#c62828", zorder=3)
        ax.annotate(f"threshold {op.threshold:.2f}, {lab}:\n{pt.automation_rate:.0%} auto-routed, {pt.auto_routed_accuracy:.0%} accurate",
                    (pt.automation_rate, pt.auto_routed_accuracy), xytext=off, textcoords="offset points",
                    fontsize=8.5, arrowprops=dict(arrowstyle="->", color="grey"))
    ax.set(xlabel="Share of tickets routed automatically", ylabel="Accuracy of auto-routed tickets",
           title="One threshold, two kinds of traffic")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(fig_dir / "routing_tradeoff.png", dpi=150); plt.close(fig)

    norm = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace("_", "\n") for c in cm.columns]
    ax.set_xticks(range(len(short)), short, fontsize=8)
    ax.set_yticks(range(len(short)), [c.replace("_", " ") for c in cm.index], fontsize=8)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if norm.values[i, j] >= 0.01:
                ax.text(j, i, f"{norm.values[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if norm.values[i, j] > 0.5 else "black")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix (familiar phrasings, row %)")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrix.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw/tickets.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
