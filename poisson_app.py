"""
Poisson Distribution Explorer
--------------------------------
A minimal Streamlit app: pick lambda (average rate), pick k (or a range),
see the probability and the distribution plot with the relevant region shaded.

Tech stack:
    streamlit  -> UI
    scipy      -> poisson math (pmf/cdf)
    matplotlib -> plotting
    numpy      -> array of k values

Run with:
    streamlit run poisson_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import poisson

st.set_page_config(page_title="Poisson Distribution Explorer", layout="centered")

st.title("Poisson Distribution Explorer")
st.write(
    "Set the average rate **λ (lambda)**, the expected number of events per interval — "
    "then choose what probability you want to compute."
)

# ── Input: lambda ─────────────────────────────────────────────────
lam = st.slider("Average rate (λ)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)

mean = lam
std = lam ** 0.5
st.caption(f"Mean (λ) = {mean:.2f}    and      Std dev(√λ) = {std:.2f}")

# Reasonable upper bound for k-inputs and the plot, based on lambda
k_max = int(lam + 4 * std) + 5

st.divider()

# ── Probability query type ───────────────────────────────────────
st.subheader("Probability calculator")

query_type = st.radio(
    "What do you want to find?",
    ["Exact: P(X = k)", "Below: P(X ≤ k)", "Above: P(X ≥ k)", "Between: P(a ≤ X ≤ b)"],
    horizontal=True,
)

result = None
shade_low, shade_high = None, None  # k-range to highlight on the plot

if query_type == "Exact: P(X = k)":
    k = st.number_input("k", min_value=0, max_value=k_max, value=min(int(lam), k_max))
    result = poisson.pmf(k, lam)
    shade_low, shade_high = k, k
    st.latex(rf"P(X = {k}) = \frac{{e^{{-\lambda}} \lambda^{{{k}}}}}{{{k}!}}")

elif query_type == "Below: P(X ≤ k)":
    k = st.number_input("k", min_value=0, max_value=k_max, value=min(int(lam), k_max))
    result = poisson.cdf(k, lam)
    shade_low, shade_high = 0, k
    st.latex(rf"P(X \leq {k}) = \sum_{{i=0}}^{{{k}}} P(X = i)")

elif query_type == "Above: P(X ≥ k)":
    k = st.number_input("k", min_value=0, max_value=k_max, value=min(int(lam), k_max))
    # P(X >= k) = 1 - P(X <= k-1) = survival function at k-1
    result = poisson.sf(k - 1, lam)
    shade_low, shade_high = k, k_max
    st.latex(rf"P(X \geq {k}) = 1 - P(X \leq {k-1})")

else:  # Between
    a, b = st.slider(
        "Range [a, b]",
        min_value=0, max_value=k_max,
        value=(max(0, int(lam) - 3), int(lam) + 3)
    )
    result = poisson.cdf(b, lam) - poisson.cdf(a - 1, lam)
    shade_low, shade_high = a, b
    st.latex(rf"P({a} \leq X \leq {b}) = P(X \leq {b}) - P(X \leq {a-1})")

st.metric("Probability", f"{result:.4f}", help=f"{result:.2%}")

st.divider()

# ── Plot: full pmf with the queried region shaded ────────────────
st.subheader("Distribution plot")

k_vals = np.arange(0, k_max + 1)
pmf_vals = poisson.pmf(k_vals, lam)

fig, ax = plt.subplots(figsize=(8, 4))

# base bars: all outcomes, muted colour
colors = [
    "#4f8ef7" if shade_low <= k <= shade_high else "#d0d4dc"
    for k in k_vals
]
ax.bar(k_vals, pmf_vals, color=colors, edgecolor="white", linewidth=0.3)

ax.axvline(mean, color="#f7a24f", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
ax.set_xlabel("k (number of events)")
ax.set_ylabel("P(X = k)")
ax.set_title(f"Poisson(λ={lam})  with  shaded region = {result:.2%}")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(
    "Blue bars mark the outcomes included in the probability you selected above; "
    "the dashed line marks the distribution's mean. "
    "Note: unlike the Binomial, Poisson has no upper limit on k, the plot is truncated "
    "at a reasonable range based on λ."
)
