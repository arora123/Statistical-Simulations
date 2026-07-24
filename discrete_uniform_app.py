"""
Discrete Uniform Distribution Explorer
--------------------------------
A minimal Streamlit app: pick the range [a, b] (all integers equally likely),
pick k (or a range), see the probability and the distribution plot with the
relevant region shaded.

Tech stack:
    streamlit  -> UI
    scipy      -> discrete uniform math (pmf/cdf) via scipy.stats.randint
    matplotlib -> plotting
    numpy      -> array of k values

Run with:
    streamlit run discrete_uniform_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import randint

st.markdown(
    "<h1 style='font-size: 34px;'>Discrete Uniform Distribution Explorer</h1>",
    unsafe_allow_html=True
)
st.write(
    "Set the lower bound **a** and upper bound **b**, every integer in [a, b] is "
    "equally likely. Then choose what probability you want to compute."
)

# ── Inputs: a and b ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("Lower bound (a)", min_value=-100, max_value=100, value=1)
with col2:
    b = st.number_input("Upper bound (b)", min_value=a, max_value=200, value=max(a, 10))

n_outcomes = b - a + 1
mean = (a + b) / 2
std = ((n_outcomes ** 2 - 1) / 12) ** 0.5

st.caption(f"Mean = {mean:.2f}    Std dev = {std:.2f}")

# scipy's randint uses a half-open interval [low, high), so high = b + 1
dist = randint(a, b + 1)

st.divider()

# ── Probability query type ───────────────────────────────────────
st.subheader("Probability calculator")

query_type = st.radio(
    "What do you want to find?",
    ["Exact: P(X = k)", "Below: P(X ≤ k)", "Above: P(X ≥ k)", "Between: P(a' ≤ X ≤ b')"],
    horizontal=True,
)

result = None
shade_low, shade_high = None, None  # k-range to highlight on the plot

if query_type == "Exact: P(X = k)":
    k = st.number_input("k", min_value=a, max_value=b, value=a)
    result = dist.pmf(k)
    shade_low, shade_high = k, k
    st.latex(rf"P(X = {k}) = \frac{{1}}{{b - a + 1}} = \frac{{1}}{{{n_outcomes}}}")

elif query_type == "Below: P(X ≤ k)":
    k = st.number_input("k", min_value=a, max_value=b, value=a)
    result = dist.cdf(k)
    shade_low, shade_high = a, k
    st.latex(rf"P(X \leq {k}) = \sum_{{i=a}}^{{{k}}} P(X = i)")

elif query_type == "Above: P(X ≥ k)":
    k = st.number_input("k", min_value=a, max_value=b, value=a)
    # P(X >= k) = 1 - P(X <= k-1) = survival function at k-1
    result = dist.sf(k - 1)
    shade_low, shade_high = k, b
    st.latex(rf"P(X \geq {k}) = 1 - P(X \leq {k-1})")

else:  # Between
    lo, hi = st.slider(
        "Range [a', b']",
        min_value=a, max_value=b,
        value=(a, b)
    )
    result = dist.cdf(hi) - dist.cdf(lo - 1)
    shade_low, shade_high = lo, hi
    st.latex(rf"P({lo} \leq X \leq {hi}) = P(X \leq {hi}) - P(X \leq {lo-1})")

st.metric("Probability", f"{result:.4f}", help=f"{result:.2%}")

st.divider()

# ── Plot: full pmf with the queried region shaded ────────────────
st.subheader("Distribution plot")

k_vals = np.arange(a, b + 1)
pmf_vals = dist.pmf(k_vals)

fig, ax = plt.subplots(figsize=(8, 4))

# base bars: all outcomes, muted colour
colors = [
    "#4f8ef7" if shade_low <= k <= shade_high else "#d0d4dc"
    for k in k_vals
]
ax.bar(k_vals, pmf_vals, color=colors, edgecolor="white", linewidth=0.3)

ax.axvline(mean, color="#f7a24f", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
ax.set_xlabel("k")
ax.set_ylabel("P(X = k)")
ax.set_title(f"Discrete Uniform(a={a}, b={b})  with shaded region = {result:.2%}")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(
    "Blue bars mark the outcomes included in the probability you selected above; "
    "the dashed line marks the distribution's mean. Every bar has the same height, "
    "that's the defining feature of a uniform distribution."
)

st.divider()

st.caption(
    "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
    "Found an issue, or interested in corporate training / speaking? "
    "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or [Email](dr.aroranisha@gmail.com)."
)
