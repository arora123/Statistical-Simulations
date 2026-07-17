"""
Binomial Distribution Explorer
--------------------------------
A minimal Streamlit app: pick n and p, pick k (or a range), see the
probability and the distribution plot with the relevant region shaded.

Tech stack:
    streamlit  -> UI
    scipy      -> binomial math (pmf/cdf)
    matplotlib -> plotting
    numpy      -> array of k values

Run with:
    streamlit run binomial_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import binom

st.title("Binomial Distribution Explorer")
st.write(
    "Set the number of trials **n** and success probability **p**, "
    "then choose what probability you want to compute."
)

# ── Inputs: n and p ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    n = st.slider("Number of trials (n)", min_value=1, max_value=200, value=20)
with col2:
    p = st.slider("Probability of success (p)", min_value=0.0, max_value=1.0, 
                  value=0.5, step=0.01)

mean = n * p
std = (n * p * (1 - p)) ** 0.5
st.caption(f"Mean (n·p) = {mean:.2f}    and      Std dev(√(n·p·(1-p))) = {std:.2f}")

st.divider()

# ── Probability query type ───────────────────────────────────────
st.header("Probability calculator")

query_type = st.radio(
    "What do you want to find?",
    ["Exact: P(X = k)", "Below: P(X ≤ k)", "Above: P(X ≥ k)", "Between: P(a ≤ X ≤ b)"],
    horizontal=True,
)

result = None
shade_low, shade_high = None, None  # k-range to highlight on the plot

if query_type == "Exact: P(X = k)":
    k = st.number_input("k", min_value=0, max_value=n, value=min(10, n))
    result = binom.pmf(k, n, p)
    shade_low, shade_high = k, k
    st.latex(rf"P(X = {k}) = \binom{{{n}}}{{{k}}} p^{{{k}}} (1-p)^{{{n-k}}}")

elif query_type == "Below: P(X ≤ k)":
    k = st.number_input("k", min_value=0, max_value=n, value=min(10, n))
    result = binom.cdf(k, n, p)
    shade_low, shade_high = 0, k
    st.latex(rf"P(X \leq {k}) = \sum_{{i=0}}^{{{k}}} P(X = i)")

elif query_type == "Above: P(X ≥ k)":
    k = st.number_input("k", min_value=0, max_value=n, value=min(10, n))
    # P(X >= k) = 1 - P(X <= k-1) = survival function at k-1
    result = binom.sf(k - 1, n, p)
    shade_low, shade_high = k, n
    st.latex(rf"P(X \geq {k}) = 1 - P(X \leq {k-1})")

else:  # Between
    a, b = st.slider("Range [a, b]", min_value=0, max_value=n, value=(max(0, n // 2 - 5), n // 2 + 5))
    result = binom.cdf(b, n, p) - binom.cdf(a - 1, n, p)
    shade_low, shade_high = a, b
    st.latex(rf"P({a} \leq X \leq {b}) = P(X \leq {b}) - P(X \leq {a-1})")

st.metric("Probability", f"{result:.4f}", help=f"{result:.2%}")

st.divider()

# ── Plot: full pmf with the queried region shaded ────────────────
st.subheader("Distribution plot")

k_vals = np.arange(0, n + 1)
pmf_vals = binom.pmf(k_vals, n, p)

fig, ax = plt.subplots(figsize=(8, 4))

# base bars: all outcomes, muted colour
colors = [
    "#4f8ef7" if shade_low <= k <= shade_high else "#d0d4dc"
    for k in k_vals
]
ax.bar(k_vals, pmf_vals, color=colors, edgecolor="white", linewidth=0.3)

ax.axvline(mean, color="#f7a24f", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
ax.set_xlabel("k (number of successes)")
ax.set_ylabel("P(X = k)")
ax.set_title(f"Binomial(n={n}, p={p})  with shaded region = {result:.2%}")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(
    "Blue bars mark the outcomes included in the probability you selected above; "
    "the dashed line marks the distribution's mean."
)
