"""
Classification Metrics Explorer: Accuracy, Precision, Recall & Beyond
--------------------------------
Given a set of actual outcomes (Y = 0/1) and predicted probabilities, this
app lets learners pick a decision cutoff (default 0.5) — turning those
probabilities into predicted classes — and then explore the standard
classification evaluation metrics one at a time: formula, computed result,
and a plain-language interpretation.

Two ways to get data in:
  1. Sample data — a synthetic, adjustable-balance dataset generated on the fly.
  2. Upload your own CSV — pick which column is the actual Y and which is
     the predicted probability; predicted class is always computed live
     from the cutoff, never uploaded directly.

The KS Statistic is treated separately from the others, since — like
correlation was for a fitted line, and probability was for a chosen cutoff
— it isn't tied to any single cutoff: it's the best possible separation a
model achieves across every threshold at once.

Tech stack (same as the other apps):
    streamlit  -> UI
    numpy      -> metric math
    matplotlib -> plotting (KS chart)
    io / csv   -> CSV upload parsing and CSV download, no pandas needed

Run with:
    streamlit run classification_metrics_app.py
"""

import csv
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Classification Metrics Explorer", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4682B4;
        color: white;
        border: 2px solid #2F4F4F;
    }
    div.stButton > button:hover {
        background-color: #5F9EA0;
        color: white;
        border: 2px solid #2F4F4F;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2E8B57;
        border: 2px solid #1F5A3A;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #3CB371;
        border: 2px solid #1F5A3A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Classification Metrics Explorer: Accuracy, Precision, "
    "Recall &amp; Beyond</h1>",
    unsafe_allow_html=True,
)

st.write(
    "A confusion matrix hides a whole family of metrics inside it, each one answering a "
    "**different question** about how the model is doing. Pick a metric below to see exactly "
    "how it's calculated and what it's telling you."
)

BLOG_URL = "https://learnerworld.tumblr.com/"

METRICS = [
    "Accuracy",
    "Misclassification Error",
    "Precision/ PPV",
    "Recall / Sensitivity / TPR",
    "Specificity / TNR",
    "F1-score",
    "KS Statistic",
]


def safe_div(a, b):
    return a / b if b > 0 else None


def compute_confusion(y, pred):
    TP = int(np.sum((y == 1) & (pred == 1)))
    TN = int(np.sum((y == 0) & (pred == 0)))
    FP = int(np.sum((y == 0) & (pred == 1)))
    FN = int(np.sum((y == 1) & (pred == 0)))
    return TP, TN, FP, FN


def compute_ks(y, p):
    pos, neg = (y == 1), (y == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    thresholds = np.sort(np.unique(np.concatenate(([0.0, 1.0], p))))
    tpr_list, fpr_list = [], []
    best_ks, best_t = -1.0, thresholds[0]
    for t in thresholds:
        pred = p >= t
        tpr = np.sum(pos & pred) / n_pos
        fpr = np.sum(neg & pred) / n_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        if tpr - fpr > best_ks:
            best_ks, best_t = tpr - fpr, t
    return {
        "ks": best_ks, "best_t": best_t,
        "thresholds": thresholds,
        "tpr": np.array(tpr_list), "fpr": np.array(fpr_list),
    }


def make_sample_data(balance_choice: str):
    rng = np.random.default_rng()
    n = 200
    pos_rate = 0.5 if balance_choice == "Balanced (~50/50)" else 0.15
    y = rng.binomial(1, pos_rate, n)
    p = np.empty(n)
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    if n_pos > 0:
        p[y == 1] = rng.beta(6, 2, n_pos)   # skewed high — a reasonably good classifier
    if n_neg > 0:
        p[y == 0] = rng.beta(2, 6, n_neg)   # skewed low
    return y, p


# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    data_source = st.radio("Data source", ["Use sample data", "Upload my own CSV"])

    y_data, p_data = None, None

    if data_source == "Use sample data":
        balance_choice = st.selectbox("Class balance", ["Balanced (~50/50)", "Imbalanced (~15% positive)"])
        needs_new_data = (
            "y_data" not in st.session_state
            or st.session_state.get("balance_choice") != balance_choice
            or st.session_state.get("data_source") != data_source
        )
        if needs_new_data:
            st.session_state.y_data, st.session_state.p_data = make_sample_data(balance_choice)
            st.session_state.balance_choice = balance_choice
            st.session_state.data_source = data_source

        if st.button("🎲 New sample (same balance)", use_container_width=True):
            st.session_state.y_data, st.session_state.p_data = make_sample_data(balance_choice)
            st.rerun()

        y_data, p_data = st.session_state.y_data, st.session_state.p_data

    else:
        uploaded = st.file_uploader("CSV with an actual-Y column and a predicted-probability column", type=["csv"])
        if uploaded is not None:
            content = uploaded.getvalue().decode("utf-8")
            reader = csv.DictReader(StringIO(content))
            rows = list(reader)
            columns = reader.fieldnames or []

            y_col = st.selectbox("Which column is the actual Y (0/1)?", columns)
            p_col = st.selectbox("Which column is the predicted probability?", columns,
                                   index=min(1, len(columns) - 1))

            try:
                y_raw = np.array([float(row[y_col]) for row in rows])
                p_raw = np.array([float(row[p_col]) for row in rows])
            except (ValueError, KeyError):
                st.error("Couldn't read one of those columns as numbers — check your column choices.")
            else:
                if not set(np.unique(y_raw)).issubset({0.0, 1.0}):
                    st.error("The actual-Y column must contain only 0s and 1s.")
                elif not np.all((p_raw >= 0) & (p_raw <= 1)):
                    st.error("The predicted-probability column must contain values between 0 and 1.")
                else:
                    y_data, p_data = y_raw, p_raw
        st.session_state.data_source = data_source

    st.divider()

    cutoff = st.number_input(
        "Decision cutoff (threshold)",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01, format="%.2f",
    )

    st.divider()

    selected_metric = st.selectbox("Metric to explore", METRICS)

    st.caption(
    "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
    "Found an issue, or interested in corporate training / speaking? "
    "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or "
    "[Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
    )

    st.divider()

    st.info(
        "🔜 **Coming next:** ROC-AUC, evaluating a classifier across every possible cutoff "
        "at once, instead of just one.",
        icon="🔜",
    )

    st.divider()

    st.caption(
        "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
        "Found an issue, or interested in corporate training / speaking? "
        "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or "
        "[Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
    )

# ── Stop here with a friendly message if there's no valid data yet ───
if y_data is None or p_data is None or len(y_data) == 0:
    st.info(" Pick sample data, or upload a CSV and choose your columns, to get started.", icon="👈")
    st.stop()

# ── Everything below depends on the chosen cutoff ─────────────────────
pred_class = (p_data >= cutoff).astype(int)
TP, TN, FP, FN = compute_confusion(y_data, pred_class)
n = len(y_data)

accuracy = safe_div(TP + TN, n)
misclass = (1 - accuracy) if accuracy is not None else None
precision = safe_div(TP, TP + FP)
recall = safe_div(TP, TP + FN)
specificity = safe_div(TN, TN + FP)
f1 = safe_div(2 * precision * recall, precision + recall) if (precision is not None and recall is not None) else None
ks_result = compute_ks(y_data, p_data)

# ── Main layout: data + confusion matrix on the left, metric deep-dive on the right ──
left_col, separator_col, right_col = st.columns([1.1, 0.02, 1.1], gap="small")
separator_col.markdown(
    "<div style='border-left: 2px solid #d0d7de; height: 100%; min-height: 420px; margin: 0 auto;'></div>",
    unsafe_allow_html=True,
)

with left_col:
    st.subheader("Let's look at the data and predictions")

    order = np.argsort(-p_data)
    table_rows = [
        {"Actual Y": int(y_data[i]), "Predicted prob.": round(float(p_data[i]), 3), "Predicted class": int(pred_class[i])}
        for i in order
    ]
    st.dataframe(table_rows, height=420, use_container_width=True, hide_index=True)
    st.caption("Scroll to see all rows, or open the table in full screen for a larger view.")

    csv_buffer = StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=table_rows[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(table_rows)
    st.download_button(
        "Download data as CSV", csv_buffer.getvalue(), "classification_predictions.csv", "text/csv",
        icon=":material/download:", use_container_width=True,
    )

    # st.divider()
    st.subheader("Confusion matrix")
    confusion_rows = [
        {"": "Actual 1", "Predicted 1": TP, "Predicted 0": FN},
        {"": "Actual 0", "Predicted 1": FP, "Predicted 0": TN},
    ]
    st.table(confusion_rows)
    st.caption(f"n = {n}  ·  TP={TP}  TN={TN}  FP={FP}  FN={FN}")

    confusion_buffer = StringIO()
    confusion_writer = csv.DictWriter(
        confusion_buffer,
        fieldnames=["Actual group", "Predicted 1", "Predicted 0"],
    )
    confusion_writer.writeheader()
    confusion_writer.writerows(
        {"Actual group": row[""], "Predicted 1": row["Predicted 1"], "Predicted 0": row["Predicted 0"]}
        for row in confusion_rows
    )
    st.download_button(
        "Download confusion matrix",
        confusion_buffer.getvalue(),
        "confusion_matrix.csv",
        "text/csv",
        icon=":material/download:",
        use_container_width=True,
    )

with right_col:
    st.subheader(f"Formula & result: {selected_metric}")

    if selected_metric == "Accuracy":
        st.metric("Accuracy", f"{accuracy:.1%}" if accuracy is not None else "n/a")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.write(f"Out of every 100 predictions, about **{accuracy*100:.0f}** were correct.")

    elif selected_metric == "Misclassification Error":
        st.metric("Misclassification Error", f"{misclass:.1%}" if misclass is not None else "n/a")
        st.latex(r"\text{Misclassification Error} = \frac{FP + FN}{TP + TN + FP + FN} = 1 - \text{Accuracy}")
        st.write(f"About **{misclass*100:.0f}** out of every 100 predictions were wrong.")

    elif selected_metric == "Precision/ PPV":
        st.metric("Precision/ PPV", f"{precision:.1%}" if precision is not None else "undefined")
        st.latex(r"\text{Precision/ PPV} = \frac{TP}{TP + FP}")
        if precision is not None:
            st.write(f"When the model predicts **positive**, it's right about **{precision*100:.0f}%** of the time.")
        else:
            st.warning("Undefined here — the model made **no positive predictions** at this cutoff.", icon="⚠️")

    elif selected_metric == "Recall / Sensitivity / TPR":
        st.metric("Recall / Sensitivity / TPR", f"{recall:.1%}" if recall is not None else "undefined")
        st.latex(r"\text{Recall / Sensitivity / TPR} = \frac{TP}{TP + FN}")
        st.caption("Recall, Sensitivity and TPR are the same metric, just different names you'll see used interchangeably.")
        if recall is not None:
            st.write(f"Of all the **actual positives**, the model correctly catches about **{recall*100:.0f}%**.")
        else:
            st.warning("Undefined here — there are **no actual positives** in this data.", icon="⚠️")

    elif selected_metric == "Specificity / TNR":
        st.metric("Specificity / TNR", f"{specificity:.1%}" if specificity is not None else "undefined")
        st.latex(r"\text{Specificity / TNR} = \frac{TN}{TN + FP}")
        if specificity is not None:
            st.write(f"Of all the **actual negatives**, the model correctly identifies about **{specificity*100:.0f}%**.")
        else:
            st.warning("Undefined here — there are **no actual negatives** in this data.", icon="⚠️")

    elif selected_metric == "F1-score":
        st.metric("F1-score", f"{f1:.3f}" if f1 is not None else "undefined")
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        if f1 is not None:
            st.write(
                "F1 blends precision and recall into one number, useful when you need to avoid "
                "**both** false positives and false negatives."
            )
        else:
            st.warning("Undefined here — precision or recall isn't defined at this cutoff.", icon="⚠️")

    else:  # KS Statistic
        if ks_result is None:
            st.warning("Can't compute KS — the data needs at least one actual 0 and one actual 1.", icon="⚠️")
        else:
            st.metric("KS Statistic", f"{ks_result['ks']:.3f}")
            st.latex(r"KS = \max_t \left( TPR(t) - FPR(t) \right)")
            st.write(
                f"Unlike the other metrics here, KS **doesn't depend on your chosen cutoff**. "
                f"It's the largest gap between the True Positive Rate and False Positive Rate "
                f"across *every* possible threshold. For this data, that gap peaks at "
                f"**{ks_result['ks']:.3f}** around threshold ≈ **{ks_result['best_t']:.2f}**. "
                "Bigger gaps mean the model separates the two classes more cleanly overall."
            )

    st.caption(f"📚 Read more about {selected_metric} on [learnerworld.tumblr.com]({BLOG_URL}).")

    st.subheader("A note on the accuracy paradox")
    st.info(
              "In many real situations, such as fraud detection, medical diagnosis, spam filters, titanic survival prediction,"
              "data is often **unbalanced** i.e., very less cases in one class and rest in other class. "
              "In such cases,accuracy can be misleading.\n\n" 
              "Read more about accuracy paradox [on my blog](https://learnerworld.tumblr.com/post/152327498485/enjoystatisticswithmebinaryclassifierperformance)."
              " Also read about other metrics like precision, recall, F1-score, AUC-ROC, etc. [here](https://learnerworld.tumblr.com/)",
            icon="⚠️",
        )
    
# ── KS chart — shown only when KS is the metric being explored ────────
if selected_metric == "KS Statistic" and ks_result is not None:
    st.divider()
    st.subheader("The KS chart")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ks_result["thresholds"], ks_result["tpr"], color="#4f8ef7", linewidth=2, label="TPR (sensitivity)")
    ax.plot(ks_result["thresholds"], ks_result["fpr"], color="#f7a24f", linewidth=2, label="FPR (1 − specificity)")
    ax.axvline(ks_result["best_t"], color="#c97ef7", linestyle="--", linewidth=1.3,
                label=f"max gap at t ≈ {ks_result['best_t']:.2f}")
    ax.axvline(cutoff, color="#888888", linestyle=":", linewidth=1.2, label=f"your cutoff = {cutoff:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    st.pyplot(fig)

    st.caption(
        "The purple dashed line marks where TPR and FPR pull furthest apart — the point of "
        "best separation. Notice it doesn't have to line up with your chosen cutoff (gray "
        "dotted line) at all."
    )

# ── All metrics at a glance ────────────────────────────────────────────
# st.divider()
st.subheader("All metrics at a glance")

glance_rows = [
    {"Metric": "Accuracy", "Value": f"{accuracy:.1%}" if accuracy is not None else "n/a"},
    {"Metric": "Misclassification Error", "Value": f"{misclass:.1%}" if misclass is not None else "n/a"},
    {"Metric": "Precision / PPV", "Value": f"{precision:.1%}" if precision is not None else "undefined"},
    {"Metric": "Recall / Sensitivity / TPR", "Value": f"{recall:.1%}" if recall is not None else "undefined"},
    {"Metric": "Specificity / TNR", "Value": f"{specificity:.1%}" if specificity is not None else "undefined"},
    {"Metric": "F1-score", "Value": f"{f1:.3f}" if f1 is not None else "undefined"},
    {"Metric": "KS Statistic", "Value": f"{ks_result['ks']:.3f}" if ks_result is not None else "n/a"},
]
st.table(glance_rows)

glance_buffer = StringIO()
glance_writer = csv.DictWriter(glance_buffer, fieldnames=["Metric", "Value"])
glance_writer.writeheader()
glance_writer.writerows(glance_rows)
st.download_button(
    "Download all metrics",
    glance_buffer.getvalue(),
    "all_classification_metrics.csv",
    "text/csv",
    icon=":material/download:",
)

st.caption(
        f"📚 For a deeper conceptual walkthrough of these metrics, visit "
        f"[learnerworld.tumblr.com]({BLOG_URL})."
    )