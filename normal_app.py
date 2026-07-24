"""
Normal Distribution Explorer
--------------------------------
A minimal Streamlit app: pick mean (μ) and std dev (σ), pick a value or
range, see the probability and the density plot with the relevant region
shaded.

Note: like the Continuous Uniform app, this is a CONTINUOUS distribution.
There is no "exact" probability query here — P(X = any single value) is
always 0 for a continuous variable, since probability is measured as AREA
under the curve, not curve height. This app offers Below / Above / Between
accordingly, plus a live z-score readout since that's the standard way
Normal probabilities are looked up and reasoned about.

Tech stack:
    streamlit  -> UI
    scipy      -> normal math (pdf/cdf) via scipy.stats.norm
    matplotlib -> plotting
    numpy      -> array of x values for the density curve

Run with:
    streamlit run normal_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# st.set_page_config(page_title="Normal Distribution Explorer", layout="centered")

st.title("Normal Distribution Explorer")
st.write(
    "Set the mean **μ (mu)** and standard deviation **σ (sigma)**, the bell curve's "
    "center and spread, then choose what probability you want to compute."
)

st.info(
    "This is a **continuous** distribution: probability is the *area* under the curve, "
    "not the height at a point. So there's no 'exact' P(X = x) query here. "
    "That value is always 0.",
    icon="ℹ️",
)

# ── Inputs: mu and sigma ──────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    mu = st.number_input("Mean (μ)", min_value=-100.0, max_value=100.0, value=0.0, step=0.5)
with col2:
    sigma = st.number_input("Std dev (σ)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)

st.caption(f"Mean (μ) = {mu:.2f}        and       Std dev (σ) = {sigma:.2f}")

dist = norm(loc=mu, scale=sigma)

# plotting range: mu ± 4 std devs comfortably covers >99.99% of the curve
x_min, x_max = mu - 4 * sigma, mu + 4 * sigma

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
z_note = ""  # z-score text shown below the result

if query_type == "Below: P(X ≤ x)":
    x = st.slider("x", min_value=float(x_min), max_value=float(x_max), 
                  value=float(mu))
    result = dist.cdf(x)
    shade_low, shade_high = x_min, x
    z = (x - mu) / sigma
    z_note = f"z = (x − μ) / σ = {z:.2f}"
    st.latex(r"P(X \leq x) = \Phi\left(\frac{x - \mu}{\sigma}\right)")

elif query_type == "Above: P(X ≥ x)":
    x = st.slider("x", min_value=float(x_min), max_value=float(x_max), value=float(mu))
    result = dist.sf(x)
    shade_low, shade_high = x, x_max
    z = (x - mu) / sigma
    z_note = f"z = (x − μ) / σ = {z:.2f}"
    st.latex(r"P(X \geq x) = 1 - \Phi\left(\frac{x - \mu}{\sigma}\right)")

else:  # Between
    x1, x2 = st.slider(
        "Range [x₁, x₂]",
        min_value=float(x_min), max_value=float(x_max),
        value=(float(mu - sigma), float(mu + sigma))
    )
    result = dist.cdf(x2) - dist.cdf(x1)
    shade_low, shade_high = x1, x2
    z1, z2 = (x1 - mu) / sigma, (x2 - mu) / sigma
    z_note = f"z₁ = {z1:.2f}    z₂ = {z2:.2f}"
    st.latex(r"P(x_1 \leq X \leq x_2) = \Phi\left(\frac{x_2 - \mu}{\sigma}\right) - \Phi\left(\frac{x_1 - \mu}{\sigma}\right)")

st.metric("Probability", f"{result:.4f}", help=f"{result:.2%}")
st.caption(z_note)

st.divider()

# ── Plot: density curve with the queried region shaded ───────────
st.subheader("Distribution plot")

x_vals = np.linspace(x_min, x_max, 500)
pdf_vals = dist.pdf(x_vals)

fig, ax = plt.subplots(figsize=(8, 4))

# base curve across the full plotting range
ax.plot(x_vals, pdf_vals, color="#d0d4dc", linewidth=1.5)

# shaded region: fill only between shade_low and shade_high
mask = (x_vals >= shade_low) & (x_vals <= shade_high)
ax.fill_between(x_vals[mask], pdf_vals[mask], color="#4f8ef7", alpha=0.85)

ax.axvline(mu, color="#f7a24f", linestyle="--", linewidth=1, 
           label=f"mean = {mu:.1f}")
ax.set_xlabel("x")
ax.set_ylabel("Density f(x)")
ax.set_ylim(bottom=0)
ax.set_title(f"Normal (μ={mu}, σ={sigma}) with shaded area = {result:.2%}")
ax.legend()
fig.tight_layout()

st.pyplot(fig)

st.caption(
    "The blue shaded area marks the probability you selected above. The dashed line marks the distribution's mean. " 
    "Probability is always measured as area under the curve, not the height  at a single point. "
)
st.divider()

st.caption(
    "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
    "Found an issue, or interested in corporate training / speaking? "
    "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or [Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
)
