"""
Covariance & Correlation Explorer
--------------------------------
Users type/paste values for two numerical variables (X and Y) and get
covariance, Pearson correlation, and a scatter plot along two built-in
example datasets that illustrate common misreadings of correlation:
a non-linear relationship (near-zero correlation despite a strong,
deterministic pattern) and a spurious correlation (driven by a hidden
confounding variable, not a real link between X and Y).

Tech stack (same as the other apps):
    streamlit  -> UI
    numpy      -> parsing / array math
    scipy      -> pearsonr (correlation + p-value)
    matplotlib -> plotting

Run with:
    streamlit run covariance_correlation_app.py
"""

import re

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import pearsonr

st.set_page_config(page_title="Covariance & Correlation Explorer", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>Covariance & Correlation Explorer</h1>",
    unsafe_allow_html=True,
)
st.write(
    "The words 'correlation' and 'dependence' are thrown around often as synonyms. "
    "Both the words seems to be pretty closer but statistically these are different."
    "Read More about [Covariance vs Correlation](https://qr.ae/pFwRW9). "
    "Paste or type values for two numerical variables and see how they relate by studying "
    "covariance, correlation, and the scatter plot."
)



# ── 1. Choose data source ──────────────────────────────────────────
st.subheader("1. Choose your data")

data_mode = st.radio(
    "Data source",
    ["Enter my own data", "Example: non-linear relationship", "Example: spurious correlation"],
    horizontal=False,
)


def parse_values(text: str):
    """Split on commas/whitespace/newlines and convert to floats.
    Returns None if any token fails to parse as a number."""
    tokens = re.split(r"[,\s]+", text.strip())
    tokens = [t for t in tokens if t != ""]
    try:
        return np.array([float(t) for t in tokens])
    except ValueError:
        return None


x_vals, y_vals = None, None
x_label, y_label = "X", "Y"

if data_mode == "Enter my own data":
    col1, col2 = st.columns(2)
    with col1:
        x_text = st.text_area(
            "Variable X: comma, space, or newline separated",
            placeholder="e.g. 2, 4, 5, 7, 8, 10",
            height=140,
        )
    with col2:
        y_text = st.text_area(
            "Variable Y: comma, space, or newline separated",
            placeholder="e.g. 3, 5, 4, 9, 8, 12",
            height=140,
        )

    if x_text.strip() and y_text.strip():
        x_vals = parse_values(x_text)
        y_vals = parse_values(y_text)

        if x_vals is None or y_vals is None:
            st.error("Couldn't parse one of the inputs — make sure every value is a plain number.")
            x_vals, y_vals = None, None
        elif len(x_vals) != len(y_vals):
            st.error(f"X has {len(x_vals)} values but Y has {len(y_vals)} values — they must match.")
            x_vals, y_vals = None, None
        elif len(x_vals) < 3:
            st.error("Enter at least 3 pairs of values for a meaningful correlation.")
            x_vals, y_vals = None, None

elif data_mode == "Example: non-linear relationship":
    rng = np.random.default_rng(seed=1)
    x_vals = rng.uniform(-10, 10, 150)
    y_vals = x_vals ** 2 + rng.normal(0, 8, 150)
    x_label, y_label = "X", "Y = X² + noise"
    st.info(
        "This data follows **Y = X² + noise** — a strong, completely deterministic pattern. "
        "Watch what Pearson correlation reports for it below.",
        icon="🧪",
    )

else:  # Spurious correlation example
    rng = np.random.default_rng(seed=7)
    confound = rng.uniform(0, 100, 150)  # the hidden driver of both X and Y
    x_vals = 2.0 * confound + rng.normal(0, 15, 150)
    y_vals = 1.5 * confound + rng.normal(0, 15, 150)
    x_label, y_label = "X (driven by hidden factor Z)", "Y (also driven by hidden factor Z)"
    st.info(
        "Neither variable here causes the other. Both are independently driven by a hidden "
        "third factor **Z** (think: something like 'time' or 'population size'). "
        "Watch how strong the correlation looks anyway.",
        icon="🧪",
    )

# st.divider()

# ── 2. Stats + plot (only once we have valid data) ─────────────────
if x_vals is not None and y_vals is not None and len(x_vals) == len(y_vals) and len(x_vals) >= 3:

    n = len(x_vals)
    covariance = np.cov(x_vals, y_vals, ddof=1)[0, 1]
    corr_r, p_value = pearsonr(x_vals, y_vals)
    r_squared = corr_r ** 2

    # st.divider()

    plot_col, divider_col, result_col = st.columns([1.3, 0.02, 1.0], gap="small")
    divider_col.markdown(
        "<div style='width: 1px; height: 420px; background-color: #d0d4dc; margin: 0 auto;'></div>",
        unsafe_allow_html=True,
    )

    with plot_col:
        st.subheader("2. Scatter plot")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x_vals, y_vals, color="#333333", alpha=0.85, s=45)

        # simple linear fit, just to visualise the linear trend Pearson r is measuring
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#4682B4",
            linewidth=1.8,
            linestyle="--",
            label=f"linear fit (r = {corr_r:.2f})",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Scatter plot with linear trend line")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with result_col:
        st.subheader("3. Results")

        metric_col1, metric_col2 = result_col.columns(2)
        metric_col1.metric("n (pairs)", f"{n}")
        metric_col1.metric("Covariance", f"{covariance:.3f}")
        metric_col2.metric("Correlation (r)", f"{corr_r:.3f}")
        metric_col2.metric("R² (r squared)", f"{r_squared:.3f}")

        result_col.caption(f"p-value (H₀: ρ = 0) (= {p_value:.4f}) tests whether the correlation is significantly different from 0")

        result_col.latex(r"\text{cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})")
        result_col.latex(r"r = \frac{\text{cov}(X, Y)}{\sigma_X \, \sigma_Y}")

        result_col.caption(
            "Covariance shows the direction of a linear relationship (positive/negative) but its "
            "size depends on the units of X and Y, which makes it hard to compare across datasets. "
            "Correlation rescales it to always fall between -1 and 1, which is why it's the more "
            "commonly reported number."
        )

    # st.divider()

    # ── Educational notes ────────────────────────────────────────────
    st.subheader("Things correlation can quietly mislead you about")

    st.warning(
        "**Correlation ≠ causation.** Two variables can be strongly correlated because a "
        "hidden third variable drives them both with no direct link between them at all. "
        "Classic example: ice cream sales and drowning incidents rise together every summer, "
        "not because one causes the other, but because both are driven by hot weather. "
        "Try the 'spurious correlation' example above, the r value looks convincing, but "
        "X never actually influences Y.",
        icon="⚠️",
    )

    st.warning(
        "**Near-zero correlation ≠ no relationship.** Pearson's r only measures *linear* "
        "association. A variable can depend on another in a very strong, completely "
        "predictable way just not a straight-line way and still score close to r = 0. "
        "Try the 'non-linear relationship' example above (Y = X²): the pattern is obvious "
        "on the scatter plot, yet r comes out near zero.",
        icon="⚠️",
    )

else:
    st.info(
        "Enter valid data above (or pick an example) to see the results and the scatter plot appear here.",
        icon="👆",
    )

# st.divider()

st.divider()

st.write(
    "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
    "Found an issue, or interested in corporate training / speaking? "
    "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or [Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
)
