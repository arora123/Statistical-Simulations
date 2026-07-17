"""
Continuous Uniform Distribution Explorer
--------------------------------
A minimal Streamlit app: pick the range [a, b] (any real number equally
likely), pick a value or range, see the probability and the density plot
with the relevant region shaded.

Note: unlike the Binomial / Poisson / Discrete Uniform apps, this is a
CONTINUOUS distribution. There is no "exact" probability query here —
P(X = any single value) is always 0 for a continuous variable, since
probability is measured as AREA under the curve, not bar height. This app
only offers Below / Above / Between accordingly.

Tech stack:
    streamlit  -> UI
    scipy      -> continuous uniform math (pdf/cdf) via scipy.stats.uniform
    matplotlib -> plotting
    numpy      -> array of x values for the density curve

Run with:
    streamlit run continuous_uniform_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import uniform

st.set_page_config(page_title="New Continuous Uniform Distribution Explorer", layout="centered")

st.markdown(
    "<h1 style='font-size: 34px;'>Discrete Uniform Distribution Explorer</h1>",
    unsafe_allow_html=True
)

st.write(
    "Set the lower bound **a** and upper bound **b**, every real number in [a, b] is "
    "equally likely. Then choose what probability you want to compute."
)

st.info(
    "This is a **continuous** distribution: probability is the *area* under the curve, "
    "not the height at a point. So there's no 'exact' P(X = x) query here. "
    "That value is always 0.",
    icon="ℹ️",
)

# ── Inputs: a and b ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    a = st.number_input("Lower bound (a)", min_value=-100.0, max_value=100.0, value=0.0, step=0.5)
with col2:
    b = st.number_input("Upper bound (b)", min_value=a + 0.01, max_value=200.0, value=max(a + 0.01, 10.0), step=0.5)

width = b - a
mean = (a + b) / 2
std = width / (12 ** 0.5)

st.caption(f"Mean = {mean:.2f}   and  Std dev = {std:.2f}")

# scipy's uniform is parameterised as loc (start) and scale (width), not (a, b) directly
dist = uniform(loc=a, scale=width)

st.divider()

# ── Probability query type ───────────────────────────────────────
st.subheader("Probability calculator")

query_type = st.radio(
    "What do you want to find?",
    ["Below: P(X ≤ x)", "Above: P(X ≥ x)", "Between: P(x₁ ≤ X ≤ x₂)"],
    horizontal=True,
)

result = None
shade_low, shade_high = None, None  # x-range to highlight on the plot

if query_type == "Below: P(X ≤ x)":
    x = st.slider("x", min_value=float(a), max_value=float(b), value=float(a + width / 2))
    result = dist.cdf(x)
    shade_low, shade_high = a, x
    st.latex(rf"P(X \leq x) = \frac{{x - a}}{{b - a}}")

elif query_type == "Above: P(X ≥ x)":
    x = st.slider("x", min_value=float(a), max_value=float(b), value=float(a + width / 2))
    result = dist.sf(x)
    shade_low, shade_high = x, b
    st.latex(rf"P(X \geq x) = 1 - P(X \leq x) = \frac{{b - x}}{{b - a}}")

else:  # Between
    x1, x2 = st.slider(
        "Range [x₁, x₂]",
        min_value=float(a), max_value=float(b),
        value=(float(a + width * 0.25), float(a + width * 0.75))
    )
    result = dist.cdf(x2) - dist.cdf(x1)
    shade_low, shade_high = x1, x2
    st.latex(rf"P(x_1 \leq X \leq x_2) = \frac{{x_2 - x_1}}{{b - a}}")

st.metric("Probability", f"{result:.4f}", help=f"{result:.2%}")

st.divider()

# ── Plot: density curve with the queried region shaded ───────────
st.subheader("Distribution plot")

x_vals = np.linspace(a, b, 500)
pdf_vals = dist.pdf(x_vals)

fig, ax = plt.subplots(figsize=(8, 4))

# base curve across the full support
ax.plot(x_vals, pdf_vals, color="#d0d4dc", linewidth=1.5)

# shaded region: fill only between shade_low and shade_high
mask = (x_vals >= shade_low) & (x_vals <= shade_high)
ax.fill_between(x_vals[mask], pdf_vals[mask], color="#4f8ef7", alpha=0.85)

ax.axvline(mean, color="#f7a24f", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
ax.set_xlabel("x")
ax.set_ylabel("Density f(x)")
ax.set_ylim(bottom=0)
ax.set_title(f"Continuous Uniform(a={a}, b={b})  with  shaded area = {result:.2%}")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(
    "The blue shaded area marks the probability you selected above with a continuous "
    "uniformdistribution. The dashed line marks the distribution's mean."
)
