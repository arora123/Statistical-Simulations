"""
Slope & Intercept Explorer: Understanding the Shape of a Line
--------------------------------
Built for learners who find the phrase "y = mx + c" intimidating. Two ways in:

  1. Choose slope & intercept  -> pick the two numbers, watch the line appear.
  2. Draw a line (pick two points) -> pick two points, the app works out
     the slope & intercept for you.

Either way, the line is drawn over a relatable backdrop dataset (or a blank
grid, if preferred), with a rise/run triangle and the y-intercept marked
directly on the plot, plus plain-language sentences translating the numbers.

Tech stack (same as the other apps):
    streamlit  -> UI
    numpy      -> backdrop data + line math
    matplotlib -> plotting

Run with:
    streamlit run slope_intercept_app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Slope & Intercept Explorer", layout="wide")

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4682B4 !important;
        color: white !important;
        border: 2px solid #2F4F4F !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        background-color: #3e7aa8 !important;
        color: white !important;
        border: 2px solid #2F4F4F !important;
    }
    div.stButton > button:focus {
        box-shadow: 0 0 0 0.2rem rgba(70, 130, 180, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Slope &amp; Intercept Explorer: Understanding the Shape of a Line</h1>",
    unsafe_allow_html=True,
)

st.write(
    "Every straight line is built from just two ingredients: how **steep** it is (the "
    "**slope**) and where it **meets y-axis** (the **intercept**). No formulas required to start,"
    " just play with both and watch what happens to the line."
)

# ── Story / dataset configurations ────────────────────────────────────
STORIES = {
    "No dataset: just the grid": dict(
            x_label="X", y_label="Y",
            x_range=(-10, 10), y_range=(-10, 10),
            slope_range=(-5.0, 5.0), slope_default=1.0,
            intercept_range=(-10.0, 10.0), intercept_default=0.0,
            true_slope=None, true_intercept=None, noise_std=None,
        ),
    "Study hours → Exam score": dict(
        x_label="Hours studied", y_label="Exam score",
        x_range=(0, 10), y_range=(0, 100),
        slope_range=(-15.0, 15.0), slope_default=5.0,
        intercept_range=(0.0, 100.0), intercept_default=40.0,
        true_slope=6.0, true_intercept=35.0, noise_std=8.0,
    ),
    "Ad spend ($100s) → Sales (units)": dict(
        x_label="Ad spend ($100s)", y_label="Sales (units)",
        x_range=(0, 20), y_range=(0, 200),
        slope_range=(-10.0, 10.0), slope_default=6.0,
        intercept_range=(-20.0, 100.0), intercept_default=20.0,
        true_slope=7.0, true_intercept=15.0, noise_std=15.0,
    ),
    "Temperature (°C) → Ice cream sales": dict(
        x_label="Temperature (°C)", y_label="Ice cream sales",
        x_range=(0, 40), y_range=(-10, 120),
        slope_range=(-5.0, 5.0), slope_default=2.5,
        intercept_range=(-20.0, 40.0), intercept_default=5.0,
        true_slope=2.3, true_intercept=8.0, noise_std=10.0,
    )
    
}


def make_backdrop(story: str):
    cfg = STORIES[story]
    if cfg["true_slope"] is None:
        return None, None
    rng = np.random.default_rng()
    x = rng.uniform(cfg["x_range"][0], cfg["x_range"][1], 18)
    y = cfg["true_slope"] * x + cfg["true_intercept"] + rng.normal(0, cfg["noise_std"], 18)
    y = np.clip(y, cfg["y_range"][0], cfg["y_range"][1])
    return x, y


def format_equation(slope, intercept):
    sign = "+" if intercept >= 0 else "\u2212"
    return f"y = {slope:.2f}x {sign} {abs(intercept):.2f}"


def describe_slope(slope, x_label, y_label, ref_scale):
    if abs(slope) < 1e-9:
        return f"**Flat line.** {y_label} stays exactly the same no matter what {x_label} is."
    steep_ratio = abs(slope) / ref_scale if ref_scale else abs(slope)
    steepness = "very steep" if steep_ratio > 1.5 else ("gently sloped" if steep_ratio < 0.5 else "moderately sloped")
    direction = "upward" if slope > 0 else "downward"
    verb = "goes up" if slope > 0 else "goes down"
    return (
        f"A **{steepness}, {direction}** line: for every extra **1 unit** of {x_label}, "
        f"{y_label} **{verb} by about {abs(slope):.2f}**."
    )


def describe_intercept(intercept, x_label, y_label):
    return (
        f"When **{x_label} = 0**, the line predicts **{y_label} = {intercept:.2f}**,"
        "that's where it crosses the vertical axis."
    )


def draw_rise_run(ax, x1, y1, x2, y2, color):
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1
    ax.plot([x1, x2], [y1, y1], linestyle=":", color=color, linewidth=1.3, zorder=4)
    ax.plot([x2, x2], [y1, y2], linestyle=":", color=color, linewidth=1.3, zorder=4)
    ax.scatter([x1, x2], [y1, y2], color=color, s=55, zorder=5, edgecolor="white", linewidth=0.6)

    dx = x2 - x1
    dy = y2 - y1

    ax.annotate(f"horizontal change = {abs(dx):.1f}", xy=((x1 + x2) / 2, y1),
                xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8,
                color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
    ax.annotate(f"vertical change = {abs(dy):.1f}", xy=(x2, (y1 + y2) / 2),
                xytext=(18, 8), textcoords="offset points", va="center", fontsize=8,
                color=color, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))


def make_tick_values(start, end):
    if end <= start:
        return []

    span = end - start
    if span == 0:
        return [float(start)]

    step_base = 10 ** np.floor(np.log10(span / 5))
    step_options = [1, 2, 5, 10]
    step = next(option * step_base for option in step_options if (option * step_base) >= (span / 8))
    start_tick = np.floor(start / step) * step
    end_tick = np.ceil(end / step) * step
    ticks = np.arange(start_tick, end_tick + step / 2, step)
    return [float(v) for v in ticks if start <= v <= end]


# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    story = st.selectbox("Pick a story", list(STORIES.keys()))
    cfg = STORIES[story]
    # Normalize to floats so every st.slider() below has matching
    # min/max/value types (Streamlit requires them to match exactly).
    cfg["x_range"] = (float(cfg["x_range"][0]), float(cfg["x_range"][1]))
    cfg["y_range"] = (float(cfg["y_range"][0]), float(cfg["y_range"][1]))

    needs_new_backdrop = (
        "backdrop_story" not in st.session_state or st.session_state.backdrop_story != story
    )
    if needs_new_backdrop:
        st.session_state.bx, st.session_state.by = make_backdrop(story)
        st.session_state.backdrop_story = story

    if cfg["true_slope"] is not None:
        if st.button("🔄 New sample points", use_container_width=True):
            st.session_state.bx, st.session_state.by = make_backdrop(story)
            st.rerun()

    st.divider()

    mode = st.radio(
        "How do you want to build your line?",
        ["🎚️ Choose slope & intercept", "✏️ Draw a line (pick two points)"],
    )

    st.divider()

    compare = False
    slope2, intercept2 = None, None

    if mode == "🎚️ Choose slope & intercept":
        slope = st.number_input(
            "Slope (m)",
            min_value=cfg["slope_range"][0],
            max_value=cfg["slope_range"][1],
            value=cfg["slope_default"],
            step=0.1,
            format="%.2f",
        )
        intercept = st.number_input(
            "Intercept (c)",
            min_value=cfg["intercept_range"][0],
            max_value=cfg["intercept_range"][1],
            value=cfg["intercept_default"],
            step=0.5,
            format="%.2f",
        )

        st.divider()
        compare = st.checkbox("🔀 Compare with a second line")
        if compare:
            second_slope_default = -cfg["slope_default"] if cfg["slope_default"] != 0 else 1.0
            second_slope_default = max(cfg["slope_range"][0], min(cfg["slope_range"][1], second_slope_default))
            second_intercept_default = cfg["intercept_default"] + (cfg["intercept_range"][1] - cfg["intercept_range"][0]) * 0.2
            second_intercept_default = max(cfg["intercept_range"][0], min(cfg["intercept_range"][1], second_intercept_default))

            slope2 = st.number_input(
                "Second line — Slope (m₂)",
                min_value=cfg["slope_range"][0],
                max_value=cfg["slope_range"][1],
                value=second_slope_default,
                step=0.1,
                format="%.2f",
            )
            intercept2 = st.number_input(
                "Second line — Intercept (c₂)",
                min_value=cfg["intercept_range"][0],
                max_value=cfg["intercept_range"][1],
                value=second_intercept_default,
                step=0.5,
                format="%.2f",
            )
        x1 = x2 = y1 = y2 = None
    else:
        x_lo, x_hi = cfg["x_range"]
        y_lo, y_hi = cfg["y_range"]
        x1_default = x_lo + 0.25 * (x_hi - x_lo)
        x2_default = x_lo + 0.75 * (x_hi - x_lo)
        y1_default = y_lo + 0.35 * (y_hi - y_lo)
        y2_default = y_lo + 0.65 * (y_hi - y_lo)

        st.caption("Point A")
        x1 = st.number_input("x₁", min_value=x_lo, max_value=x_hi, value=x1_default, step=0.5, format="%.2f", key="x1")
        y1 = st.number_input("y₁", min_value=y_lo, max_value=y_hi, value=y1_default, step=0.5, format="%.2f", key="y1")
        st.caption("Point B")
        x2 = st.number_input("x₂", min_value=x_lo, max_value=x_hi, value=x2_default, step=0.5, format="%.2f", key="x2")
        y2 = st.number_input("y₂", min_value=y_lo, max_value=y_hi, value=y2_default, step=0.5, format="%.2f", key="y2")

        slope, intercept = None, None
        if x1 == x2:
            st.error("These two points sit on a vertical line — it has no slope in the y = mx + c sense. Move one sideways.")
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

    st.divider()
    st.caption(
        "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
        "Found an issue, or interested in corporate training / speaking? "
        "Reach out at [LinkedIn](https://www.linkedin.com/in/drnishaarora/) or "
        "[Email](mailto:dr.aroranisha@gmail.com?subject=Hello%20Nisha&body=I%20saw%20your%20Streamlit%20app...)"
    )

# ── Main layout ─────────────────────────────────────────────────────
left_col, separator_col, right_col = st.columns([1.25, 0.02, 0.95], gap="small")
separator_col.markdown(
    "<div style='border-left: 2px solid #d0d7de; height: 100%; min-height: 420px; margin: 0 auto;'></div>",
    unsafe_allow_html=True,
)

with left_col:
    st.subheader("Let's see the line")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if st.session_state.bx is not None:
        ax.scatter(st.session_state.bx, st.session_state.by, color="#d0d4dc",
                   edgecolor="white", linewidth=0.4, s=40, zorder=2, label=story)

    x_lo, x_hi = cfg["x_range"]
    y_lo, y_hi = cfg["y_range"]
    ref_scale = (y_hi - y_lo) / (x_hi - x_lo)

    if slope is not None:
        x_line = np.array([x_lo, x_hi])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#4682B4", linewidth=1.8, zorder=3, label="your line")

        if mode.startswith("🎚️"):
            rx1, rx2 = x_lo + 0.3 * (x_hi - x_lo), x_lo + 0.7 * (x_hi - x_lo)
            draw_rise_run(ax, rx1, slope * rx1 + intercept, rx2, slope * rx2 + intercept, "#4f8ef7")
        else:
            draw_rise_run(ax, x1, y1, x2, y2, "#4f8ef7")
            ax.scatter([x1], [y1], color="#4682B4", s=55, zorder=6, edgecolor="white", linewidth=0.6)
            ax.scatter([x2], [y2], color="#4682B4", s=55, zorder=6, edgecolor="white", linewidth=0.6)
            ax.annotate("A", (x1, y1), xytext=(8, 8), textcoords="offset points", fontsize=9,
                        color="#2b2b2b", weight="bold")
            ax.annotate("B", (x2, y2), xytext=(8, 8), textcoords="offset points", fontsize=9,
                        color="#2b2b2b", weight="bold")

        if x_lo <= 0 <= x_hi:
            ax.scatter([0], [intercept], color="#f7a24f", s=55, zorder=6, edgecolor="white", linewidth=0.6)

        if compare and slope2 is not None:
            y_line2 = slope2 * x_line + intercept2
            ax.plot(x_line, y_line2, color="#34d4a0", linewidth=2.2, linestyle="--", zorder=3, label="second line")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_axisbelow(False)
    ax.grid(False)

    # Put both axes through the origin and keep the axis labels on those lines.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1f1f1f")
    ax.spines["bottom"].set_color("#1f1f1f")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    if x_lo < 0 < x_hi:
        ax.spines["bottom"].set_position(("data", 0))
    if y_lo < 0 < y_hi:
        ax.spines["left"].set_position(("data", 0))

    xticks = make_tick_values(x_lo, x_hi)
    yticks = make_tick_values(y_lo, y_hi)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(axis="both", which="both", length=5, width=1.0, colors="#2b2b2b", labelsize=8)

    ax.set_xlabel(cfg["x_label"])
    ax.set_ylabel(cfg["y_label"])
    # ax.set_title(story)
    fig.tight_layout()
    st.pyplot(fig)

    if st.session_state.bx is not None:
        st.caption(
            "The dots are realistic example data for context this app isn't about fitting "
            "them perfectly. It's about seeing what slope and intercept do to a line. \n"
            "To understand best fit line check out the [Linear Regression Explorer](https://enjoy-stats-best-fit-line.streamlit.app/) app."
        )

with right_col:
    st.subheader("What your line says")

    if slope is None:
        st.info("Fix the vertical-line issue on the left to see the numbers here.", icon="👈")
    else:
        rc1, rc2 = st.columns(2)
        rc1.metric("Slope (m)", f"{slope:.2f}")
        rc2.metric("Intercept (c)", f"{intercept:.2f}")

        st.latex(r"Y = mX + c")

        st.latex(format_equation(slope, intercept).replace("\u2212", "-"))

        st.write(describe_slope(slope, cfg["x_label"], cfg["y_label"], ref_scale))
        st.write(describe_intercept(intercept, cfg["x_label"], cfg["y_label"]))

        st.info(
            "**In simple words:** The intercept is the value of Y when X = 0.\n\n"
            "The slope is the rate of change of Y with respect to X, it tells you how much Y changes for each unit of X."
            "Mathematicians call this rise over run, and it is exactly what the little dotted triangle on the plot is showing you.",
            icon="✅",
        )

        st.latex(r"\text{slope} = \frac{\text{rise}}{\text{run}} = \frac{y_2 - y_1}{x_2 - x_1}")

        if compare and slope2 is not None:
                st.divider()
                st.write(f"**Second line:** {format_equation(slope2, intercept2).replace(chr(8722), '- ')}")
                if abs(slope - slope2) < 1e-9:
                    st.info(
                        "Same slope, different intercept → these two lines are **parallel**: "
                        "they never cross, no matter how far you extend them.",
                        icon="📐",
                    )
                else:
                    x_cross = (intercept2 - intercept) / (slope - slope2)
                    y_cross = slope * x_cross + intercept
                    if x_lo <= x_cross <= x_hi:
                        st.info(
                            f"Different slopes → these two lines **cross** at "
                            f"X ≈ {x_cross:.2f}, Y ≈ {y_cross:.2f}.",
                            icon="📐",
                        )
                    else:
                        st.info(
                            f"Different slopes → these lines do cross eventually, at X ≈ {x_cross:.2f}, "
                            "which is outside the range shown here.",
                            icon="📐",
                        )

        st.info(
                "**Note:** This is Slope-Intercept Form of a linear equation, "
                "which is useful to understand how the data points relate to each other.\n\n"
                "Linear Regression, uses this form to find the best fit line for a given dataset.\n\n"
                "There are other forms of linear equations as well, like Point-Slope Form, Two-Point Form, Standard Form, etc. but Slope-Intercept Form is the most widely used.",
                icon="💡",
                )
