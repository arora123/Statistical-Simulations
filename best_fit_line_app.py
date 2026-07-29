"""
Best Fit Line Explorer — Many Lines Are Feasible, Only One Minimizes Cost
--------------------------------
A sample of two linearly-related numerical variables is shown as a scatter
plot. Each click on "Try a candidate line" draws a new, differently-colored
line through the cloud of points, along with its slope, intercept, and cost
(sum of squared errors). After trying a few, click "Show the Best Fit Line"
to reveal the one line (in black) that mathematically minimizes that cost —
the Ordinary Least Squares (OLS) solution.

Tech stack (same as the other apps):
    streamlit  -> UI
    numpy      -> random data + candidate lines + OLS fit (polyfit)
    scipy      -> pearsonr (correlation, a property of the data, not the line)
    matplotlib -> plotting

Run with:
    streamlit run best_fit_line_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import pearsonr

st.set_page_config(page_title="Best Fit Line Explorer", layout="wide")

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

# st.title("Finding the Best Fit Line: A Regression Explorer")

st.markdown(
    "<h1 style='text-align: center;'>Finding the Best Fit Line: A Regression"
    " Explorer</h1>",
    unsafe_allow_html=True,
)

st.write(
    "Lots of lines could plausibly pass through a cloud of points. But only **one** line "
    "minimizes the total squared distance between itself and every point, that's the "
    "*best fit* line. Let's find it by trial and error first, then see the exact answer."
)

CANDIDATE_COLORS = ["#f7a24f", "#c97ef7", "#f7614f", "#f7d24f", "#34d4a0", "#7ef7e0", "#ff8fab", "#9fa8da"]
MAX_CANDIDATES = 8


def sse_cost(x, y, slope, intercept):
    """Sum of squared errors — the cost function OLS minimizes."""
    predicted = slope * x + intercept
    return float(np.sum((y - predicted) ** 2))


def make_dataset():
    rng = np.random.default_rng()
    x = rng.uniform(0, 10, 60)
    true_slope = rng.uniform(1.5, 4)
    true_intercept = rng.uniform(-5, 10)
    noise = rng.normal(0, 4, 60)
    y = true_slope * x + true_intercept + noise
    return x, y


# ── Session state setup ─────────────────────────────────────────────
if "x_data" not in st.session_state:
    st.session_state.x_data, st.session_state.y_data = make_dataset()
    st.session_state.candidates = []       # list of dicts: slope, intercept, cost, color
    st.session_state.show_best_fit = False

x_data = st.session_state.x_data
y_data = st.session_state.y_data

# st.divider()

# ── Sidebar controls ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    if st.button("🎲 New dataset", use_container_width=True):
        st.session_state.x_data, st.session_state.y_data = make_dataset()
        st.session_state.candidates = []
        st.session_state.show_best_fit = False
        st.rerun()

    # st.divider()

    add_disabled = len(st.session_state.candidates) >= MAX_CANDIDATES
    if st.button("➕ Try a candidate line", use_container_width=True, disabled=add_disabled):
        # Base the random range loosely around the data's own OLS fit so
        # candidates look plausible rather than wildly off-screen — but the
        # OLS values themselves are never shown to the user at this stage.
        ols_slope, ols_intercept = np.polyfit(x_data, y_data, 1)
        rng = np.random.default_rng()
        cand_slope = ols_slope + rng.uniform(-2.5, 2.5)
        cand_intercept = ols_intercept + rng.uniform(-10, 10)
        cost = sse_cost(x_data, y_data, cand_slope, cand_intercept)
        color = CANDIDATE_COLORS[len(st.session_state.candidates) % len(CANDIDATE_COLORS)]
        st.session_state.candidates.append({
            "slope": cand_slope, "intercept": cand_intercept, "cost": cost, "color": color
        })
        st.rerun()

    st.caption("👉 On the right, observe the slope, intercept, and cost of every line you've tried so far.")
            
    if add_disabled:
        st.caption(
            f"That's {MAX_CANDIDATES} guesses. Plenty right? Each new line is drawn in a different color, so you can see how they compare."
        )
        st.write("👇 Click **Show the Best Fit Line** to reveal the one line that mathematically minimizes the cost function (sum of squared errors).")

    st.divider()

    if st.button("⭐ Show the Best Fit Line", use_container_width=True, type="primary"):
        st.session_state.show_best_fit = True
        st.rerun()

    st.divider()
    st.caption(
        "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
        "Found an issue, or interested in corporate training / speaking? "
        "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or [Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
    )

# ── Main layout: scatter plot on the left, data properties + cost comparison on the right ──
left_col, separator_col, right_col = st.columns([1.25, 0.02, 0.95], gap="small")

separator_col.markdown(
    "<div style='border-left: 2px solid #d0d7de; height: 100%; min-height: 420px; margin: 0 auto;'></div>",
    unsafe_allow_html=True,
)

with left_col:
    st.subheader("Let's visualize the data and candidate line(s)")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(x_data, y_data, color="#4f8ef7", alpha=0.65, edgecolor="white", linewidth=0.4, s=45, zorder=3, label="data")

    x_line = np.linspace(x_data.min(), x_data.max(), 100)

    for i, cand in enumerate(st.session_state.candidates, start=1):
        y_line = cand["slope"] * x_line + cand["intercept"]
        ax.plot(x_line, y_line, color=cand["color"], linewidth=1.6, alpha=0.9,
                 label=f"Candidate {i}  (cost = {cand['cost']:,.0f})")

    if st.session_state.show_best_fit:
        best_slope, best_intercept = np.polyfit(x_data, y_data, 1)
        best_cost = sse_cost(x_data, y_data, best_slope, best_intercept)
        y_best = best_slope * x_line + best_intercept
        ax.plot(x_line, y_best, color="#000000", linewidth=3, zorder=4,
                 label=f"Best Fit (OLS)  (cost = {best_cost:,.0f})")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Scatter plot with candidate lines" + (" and the best fit line" if st.session_state.show_best_fit else ""))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

with right_col:
    corr_r, _ = pearsonr(x_data, y_data)
    st.subheader("Let's take a look at the data's properties")
    st.metric("Correlation of X and Y (a property of the data, not of any single line)", f"{corr_r:.3f}")

    # st.divider()

    if st.session_state.candidates or st.session_state.show_best_fit:
        st.subheader("Coefficients and cost of every line tried")

        rows = []
        bar_labels, bar_costs, bar_colors = [], [], []
        for i, cand in enumerate(st.session_state.candidates, start=1):
            rows.append({
                "Line": f"Candidate {i}",
                "Slope": round(cand["slope"], 3),
                "Intercept": round(cand["intercept"], 3),
                "Cost (SSE)": round(cand["cost"], 1),
            })
            bar_labels.append(f"#{i}")
            bar_costs.append(cand["cost"])
            bar_colors.append(cand["color"])

        if st.session_state.show_best_fit:
            best_slope, best_intercept = np.polyfit(x_data, y_data, 1)
            best_cost = sse_cost(x_data, y_data, best_slope, best_intercept)
            rows.append({
                "Line": "Best Fit (OLS)",
                "Slope": round(best_slope, 3),
                "Intercept": round(best_intercept, 3),
                "Cost (SSE)": round(best_cost, 1),
            })
            bar_labels.append("Best Fit")
            bar_costs.append(best_cost)
            bar_colors.append("#000000")

        st.table(rows)

        fig2, ax2 = plt.subplots(figsize=(9, 3.2))
        ax2.bar(bar_labels, bar_costs, color=bar_colors, edgecolor="white", linewidth=0.6)
        ax2.set_ylabel("Cost (SSE)")
        ax2.set_title("Cost comparison of candidate lines and the best fit line")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

        if st.session_state.show_best_fit:
            st.success(
                "The **Best Fit (OLS)** bar is the shortest. "
                "Ordinary Least Squares solves directly for the one slope and intercept that "
                "minimize this exact cost function, so no other line, random or carefully chosen, "
                "can ever score lower on this dataset.",
                icon="✅",
            )
    else:
        st.info("Click **➕ Try a candidate line** in the sidebar to draw your first guess.", icon="👈")

# st.divider()

st.write(
    "**Ordinary Least Squares** finds the slope (m) and intercept (c) that minimize this sum "
    "(Sum of squared errors), called **Cost Function**:"
)

st.latex(r"\text{Cost (SSE)} = \sum_{i=1}^{n} \left(y_i - (m x_i + c)\right)^2")

