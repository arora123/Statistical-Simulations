"""
Logistic Regression Explorer: From Probability to Class Prediction
--------------------------------
A synthetic binary outcome (Y = 0/1) is generated from a single continuous
predictor X. A logistic regression is fitted to the data, producing the
classic S-shaped probability curve. Learners choose a decision cutoff
(default 0.5) — the point on the S-curve where a predicted probability
gets converted into a predicted class — and watch predictions, accuracy,
and misclassification change as that line moves, without the underlying
curve itself ever changing.

An optional "Imbalanced" dataset mode is included specifically to
demonstrate the accuracy paradox: with a rare positive class, a model can
score high accuracy while doing barely better than just guessing the
majority class every time.

Tech stack (same as the other apps):
    streamlit  -> UI
    numpy      -> data generation, sigmoid/logit math
    scipy      -> minimize (fits the logistic regression via max likelihood)
    matplotlib -> plotting

Run with:
    streamlit run logistic_regression_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize
from io import StringIO
import csv

st.set_page_config(page_title="Logistic Regression Explorer", layout="wide")

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
    "<h1 style='text-align: center;'>Logistic Regression Explorer: From"
    " Probability to Class Prediction</h1>",
    unsafe_allow_html=True,
)

st.write(
    "Logistic regression predicts conditional **probability** that (Y = 1) given a predictor (X). "
    "Class prediction (into a 0 or 1) requires picking a **cutoff** (default 0.5), " 
    " the point on the S-curve where a predicted probability gets converted into a predicted class. "
    "Move it below and watch what happens."
)

TP_COLOR, FN_COLOR = "#4f8ef7", "#f7a24f"   # by PREDICTED class: predicted 1 vs predicted 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def neg_log_likelihood(params, x, y):
    b0, b1 = params
    p = sigmoid(b0 + b1 * x)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(x, y):
    res = minimize(neg_log_likelihood, x0=[0.0, 0.0], args=(x, y), method="BFGS")
    return res.x  # b0, b1


def make_dataset(balance_choice: str):
    rng = np.random.default_rng()
    x = rng.uniform(-6, 6, 150)
    if balance_choice == "Balanced (~50/50)":
        true_b0, true_b1 = 0.0, 1.2
    else:  # Imbalanced — tuned so the positive class lands around ~10%
        true_b0, true_b1 = -3.0, 0.4
    p_true = sigmoid(true_b0 + true_b1 * x)
    y = rng.binomial(1, p_true)
    jitter = rng.uniform(-0.045, 0.045, 150)  # fixed per-point visual jitter, stored so it doesn't reshuffle on rerun
    return x, y, jitter


# ── Session state setup ─────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    balance_choice = st.selectbox("Class balance", ["Balanced (~50/50)", "Imbalanced (~10% positive)"])

    needs_new_data = (
        "x_data" not in st.session_state
        or st.session_state.get("balance_choice") != balance_choice
    )
    if needs_new_data:
        st.session_state.x_data, st.session_state.y_data, st.session_state.jitter = make_dataset(balance_choice)
        st.session_state.balance_choice = balance_choice

    if st.button("🎲 New sample (same balance)", use_container_width=True):
        st.session_state.x_data, st.session_state.y_data, st.session_state.jitter = make_dataset(balance_choice)
        st.rerun()

    st.caption("👉 On the right, watch each point's predicted class and the accuracy numbers update.")

    st.divider()

    cutoff = st.number_input(
        "Decision cutoff (threshold)",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        format="%.2f",
    )

    st.write(
        "✅ Try changing the cutoff toward 0 or 1 on the **Imbalanced** dataset to see the "
        "accuracy paradox in action."
    )

    st.divider()

    st.caption(
        "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
        "Found an issue, or interested in corporate training / speaking? "
        "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or "
        "[Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
    )

x_data = st.session_state.x_data
y_data = st.session_state.y_data
jitter = st.session_state.jitter

# ── Fit the model (depends only on the data, never on the cutoff) ────
b0_hat, b1_hat = fit_logistic(x_data, y_data)
p_hat = sigmoid(b0_hat + b1_hat * x_data)

# ── Everything below this line depends on the chosen cutoff ──────────
pred_class = (p_hat >= cutoff).astype(int)
accuracy = float(np.mean(pred_class == y_data))
misclass = 1 - accuracy

TP = int(np.sum((y_data == 1) & (pred_class == 1)))
TN = int(np.sum((y_data == 0) & (pred_class == 0)))
FP = int(np.sum((y_data == 0) & (pred_class == 1)))
FN = int(np.sum((y_data == 1) & (pred_class == 0)))

baseline_majority_class = 0 if np.mean(y_data == 0) >= 0.5 else 1
baseline_accuracy = max(np.mean(y_data == 0), np.mean(y_data == 1))

# ── Main layout: S-curve on the left, data + metrics on the right ────
left_col, separator_col, right_col = st.columns([1.25, 0.02, 0.95], gap="small")
separator_col.markdown(
    "<div style='border-left: 2px solid #d0d7de; height: 100%; min-height: 420px; margin: 0 auto;'></div>",
    unsafe_allow_html=True,
)

with left_col:
    st.subheader("Let's visualize the S-shaped curve and the cutoff")

    fig, ax = plt.subplots(figsize=(6, 5.5))

    x_line = np.linspace(x_data.min(), x_data.max(), 300)
    y_curve = sigmoid(b0_hat + b1_hat * x_line)
    ax.plot(x_line, y_curve, color="#333333", linewidth=2, zorder=2)

    # Point color shows predicted class; vertical position shows actual class.
    groups = [
        (pred_class == 1, TP_COLOR, "predicted 1"),
        (pred_class == 0, FN_COLOR, "predicted 0"),
    ]
    for mask, color, label in groups:
        if mask.any():
            ax.scatter(x_data[mask], y_data[mask].astype(float) + jitter[mask],
                       color=color, s=45, alpha=0.8, zorder=3, label=label)

    ax.axhline(cutoff, color="#c97ef7", linestyle="--", linewidth=1.4, zorder=1)
    ax.text(
        0.99,
        cutoff,
        f"cutoff = {cutoff:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#8b4fa3",
        fontsize=8,
    )

    ax.set_xlabel("X (predictor)")
    ax.set_ylabel("Predicted probability  P(Y = 1 | X)")
    ax.set_ylim(-0.15, 1.15)
    # ax.set_title(f"Logistic fit at cutoff = {cutoff:.2f}")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        fontsize=7.5,
        framealpha=0.9,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    st.pyplot(fig)

    st.write(
        "**Logistic regression** models the *log-odds* of the outcome as a straight line in X, "
        )
    st.latex(r"logit(P(Y=1 \mid X)) = b_0 + b_1 X")
    st.write(
            "**This model equation can be written in terms of probability** by applying the inverse logit (sigmoid) function:"
            )
    st.latex(r"P(Y=1 \mid X) = \frac{1}{1 + e^{-(b_0 + b_1 X)}}")

    st.write(
        "The cutoff you choose is a separate decision layered on top of that probability, "
        "not part of what the model itself estimates."
    )

    st.caption(
        f"For the given data, **Fitted model**:  logit(p) = {b0_hat:.2f} + {b1_hat:.2f} · X. \n\n"
        "This curve and every point's probability stay fixed as you move the cutoff, "
        "only which side of the line counts as 'predicted 1' changes."
    )



with right_col:
    st.subheader("Let's look at the data and predictions")

    n = len(y_data)
    pos_rate = float(np.mean(y_data == 1))
    m1, m2 = st.columns(2)
    m1.metric("n (points)", f"{n}")
    m2.metric("Actual positive rate", f"{pos_rate:.1%}")

    order = np.argsort(-p_hat)
    table_rows = [
        {
            "Actual Y": int(y_data[i]),
            "Predicted prob.": round(float(p_hat[i]), 3),
            "Predicted class": int(pred_class[i]),
        }
        for i in order
    ]
    st.dataframe(table_rows, height=260, use_container_width=True, hide_index=True)
    st.caption("Scroll to see all rows, or open the table in full screen for a larger view.")
    csv_buffer = StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=table_rows[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(table_rows)
    st.download_button(
        "Download data as CSV",
        csv_buffer.getvalue(),
        "logistic_predictions.csv",
        "text/csv",
        icon=":material/download:",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Accuracy & misclassification")

    a1, a2 = st.columns(2)
    a1.metric("Accuracy", f"{accuracy:.1%}")
    a2.metric("Misclassification rate", f"{misclass:.1%}")

    st.table([
        {"": "Actual 1", "Predicted 1": TP, "Predicted 0": FN},
        {"": "Actual 0", "Predicted 1": FP, "Predicted 0": TN},
    ])

    st.divider()
    st.subheader("A note on the accuracy paradox")

    st.info(
          "In many cases, such as fraud detection, medical diagnosis, spam filters, titanic survival prediction,"
          "data is often **unbalanced** i.e., very less cases in one class and rest in other class. "
          "In such cases,accuracy can be misleading.\n\n" 
          "Read more about accuracy paradox [on my blog](https://learnerworld.tumblr.com/post/152327498485/enjoystatisticswithmebinaryclassifierperformance)."
          " Also read about other metrics like precision, recall, F1-score, AUC-ROC, etc. [here](https://learnerworld.tumblr.com/)",
        icon="⚠️",
    )

    st.subheader("Bonus material on the topic")
    st.info(
        "**Researchers/SPSS users** can refer to my slide-deck on logistic regression [here](https://www.slideshare.net/slideshow/7-logistics-regression-using-spss/244366358)",
        icon="💡"
    )


