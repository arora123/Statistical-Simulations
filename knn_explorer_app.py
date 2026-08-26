"""
KNN Explorer: Classification & Regression by Nearest Neighbors
--------------------------------
K-Nearest Neighbors makes a prediction for a new point by looking at the
K closest points already seen, and:
  - Classification -> voting for the most common class among them
  - Regression     -> averaging their values

This app shows all three flavors learners usually meet first:
  1. Binary classification (2 classes)
  2. Multiclass classification (3 classes)
  3. Regression (a continuous target)

Move the query point around, change K, and watch the shaded decision
regions, the highlighted neighbors, and the prediction update together.

Tech stack:
    streamlit  -> UI
    numpy      -> distance math, vectorized grid predictions
    matplotlib -> plotting (decision regions, neighbor highlighting)
    collections.Counter -> majority vote counting
    io / csv   -> CSV download of the neighbor table, no pandas needed

Run with:
    streamlit run knn_explorer_app.py
"""

import csv
from collections import Counter
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

st.set_page_config(page_title="KNN Explorer", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

    .creator-links {
        display: flex;
        justify-content: center;
        gap: 1.25rem;
        margin: 0.25rem 0 1.25rem;
    }
    .creator-links a {
        color: #000000;
        font-size: 1.35rem;
        text-decoration: none;
    }
    .creator-links a:hover {
        color: #2E8B57;
    }

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
    "<h1 style='text-align: center;'>KNN Explorer: Classification &amp; Regression by Nearest Neighbors</h1>",
    unsafe_allow_html=True,
)


st.write(
    "K-Nearest Neighbors makes no assumptions about a formula (It's a non-parametric method) it just looks at whichever "
    "**K points are closest** to the one you're asking about, and borrows their answer: "
    "a **vote** for classification, an **average** for regression."
)

PLOT_RANGE = (-7.0, 7.0)
CLASS_COLORS = ["#4f8ef7", "#f7a24f", "#34d4a0"]
CLASS_COLORS_LIGHT = ["#dce8fd", "#fde6cf", "#d6f3e7"]


def make_binary(rng):
    n = 40
    c0 = rng.normal([-2, -1], 1.3, (n, 2))
    c1 = rng.normal([2, 1.5], 1.3, (n, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * n + [1] * n)
    return X, y


def make_multiclass(rng):
    n = 30
    c0 = rng.normal([-3, -2], 1.3, (n, 2))
    c1 = rng.normal([3, -2], 1.3, (n, 2))
    c2 = rng.normal([0, 3], 1.3, (n, 2))
    X = np.vstack([c0, c1, c2])
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


def true_z(x1, x2):
    bump1 = 12 * np.exp(-((x1 - 2) ** 2 + (x2 - 2) ** 2) / 10)
    bump2 = 8 * np.exp(-((x1 + 3) ** 2 + (x2 + 2) ** 2) / 8)
    return bump1 + bump2


def make_regression(rng):
    n = 80
    X = rng.uniform(PLOT_RANGE[0] + 1, PLOT_RANGE[1] - 1, (n, 2))
    z = true_z(X[:, 0], X[:, 1]) + rng.normal(0, 1.5, n)
    return X, z


def make_dataset(mode: str):
    rng = np.random.default_rng()
    if mode == "Binary Classification":
        return make_binary(rng)
    elif mode == "Multiclass Classification":
        return make_multiclass(rng)
    else:
        return make_regression(rng)


def knn_predict_point(train_X, train_y, query, k, is_regression):
    dists = np.sqrt(np.sum((train_X - query) ** 2, axis=1))
    idx = np.argsort(dists)[:k]
    neighbor_labels = train_y[idx]
    neighbor_dists = dists[idx]
    if is_regression:
        pred = float(neighbor_labels.mean())
    else:
        pred = np.bincount(neighbor_labels.astype(int)).argmax()
    return pred, idx, neighbor_dists, neighbor_labels


def knn_predict_grid(train_X, train_y, grid_points, k, is_regression):
    diffs = grid_points[:, None, :] - train_X[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    idx = np.argsort(dists, axis=1)[:, :k]
    neighbor_labels = train_y[idx]
    if is_regression:
        return neighbor_labels.mean(axis=1)
    preds = np.empty(len(grid_points), dtype=int)
    for i, row in enumerate(neighbor_labels):
        preds[i] = np.bincount(row.astype(int)).argmax()
    return preds


# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")

    mode = st.selectbox("Choose a task", ["Binary Classification", "Multiclass Classification", "Regression"])
    is_regression = (mode == "Regression")

    needs_new_data = (
        "train_X" not in st.session_state or st.session_state.get("mode") != mode
    )
    if needs_new_data:
        st.session_state.train_X, st.session_state.train_y = make_dataset(mode)
        st.session_state.mode = mode

    if st.button("🎲 New sample points", use_container_width=True):
        st.session_state.train_X, st.session_state.train_y = make_dataset(mode)
        st.rerun()

    k = st.number_input("Number of neighbors (K)", min_value=1, max_value=25, value=5, step=1)

    st.caption("Query point")
    qx = st.number_input("X₁", min_value=-7.0, max_value=7.0, value=0.0, step=0.1)
    qy = st.number_input("X₂", min_value=-7.0, max_value=7.0, value=0.0, step=0.1)
    
    show_regions = st.checkbox("Show decision regions", value=True)

    st.divider()

    st.caption(
        "Built by Dr. Nisha Arora, Analytics, Data Science & AI trainer. "
        "Found an issue, or interested in corporate training / speaking? Reach out at ")

    st.markdown(
        "<div class='creator-links'>"
        "<a href='https://www.linkedin.com/in/drnishaarora/' target='_blank' "
        "title='LinkedIn' aria-label='LinkedIn'><i class='fa-brands fa-linkedin'></i></a>"
        "<a href='https://learnerworld.tumblr.com/' target='_blank' "
        "title='Tumblr blog' aria-label='Tumblr blog'><i class='fa-brands fa-tumblr'></i></a>"
        "<a href='https://www.youtube.com/@DrNishaArora' target='_blank' "
        "title='YouTube' aria-label='YouTube'><i class='fa-brands fa-youtube'></i></a>"
        "<a href='https://arora123.github.io' target='_blank' "
        "title='Website' aria-label='Website'><i class='fa-solid fa-globe'></i></a>"
        "<a href='mailto:dr.aroranisha@gmail.com' title='Email' aria-label='Email'>"
        "<i class='fa-solid fa-envelope'></i></a>"
        "</div>",
        unsafe_allow_html=True,
    )

train_X = st.session_state.train_X
train_y = st.session_state.train_y
query = np.array([qx, qy])

k = min(k, len(train_X))  # safety guard, though the slider max already respects dataset size
n_classes = len(np.unique(train_y)) if not is_regression else None

pred, nbr_idx, nbr_dists, nbr_labels = knn_predict_point(train_X, train_y, query, k, is_regression)

# ── Main layout: plot on the left, prediction + neighbor table on the right ──
left_col, separator_col, right_col = st.columns([1.25, 0.02, 0.95], gap="small")
separator_col.markdown(
    "<div style='border-left: 2px solid #d0d7de; height: 100%; min-height: 420px; margin: 0 auto;'></div>",
    unsafe_allow_html=True,
)

with left_col:
    st.subheader("Let's see the neighborhood")

    fig, ax = plt.subplots(figsize=(6.2, 5.8))

    if show_regions:
        res = 60
        x_lin = np.linspace(PLOT_RANGE[0], PLOT_RANGE[1], res)
        y_lin = np.linspace(PLOT_RANGE[0], PLOT_RANGE[1], res)
        xx, yy = np.meshgrid(x_lin, y_lin)
        grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
        grid_preds = knn_predict_grid(train_X, train_y, grid_pts, k, is_regression)

        if is_regression:
            im = ax.imshow(grid_preds.reshape(xx.shape), extent=[*PLOT_RANGE, *PLOT_RANGE],
                            origin="lower", cmap="viridis", alpha=0.75, aspect="auto", zorder=1)
            fig.colorbar(im, ax=ax, label="Predicted value", shrink=0.8)
        else:
            cmap = ListedColormap(CLASS_COLORS_LIGHT[:n_classes])
            ax.imshow(grid_preds.reshape(xx.shape), extent=[*PLOT_RANGE, *PLOT_RANGE],
                       origin="lower", cmap=cmap, alpha=0.75, aspect="auto", zorder=1,
                       vmin=0, vmax=n_classes - 1)

    if is_regression:
        ax.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap="viridis",
                   edgecolor="white", linewidth=0.5, s=45, zorder=2)
    else:
        for c in range(n_classes):
            mask = train_y == c
            ax.scatter(train_X[mask, 0], train_X[mask, 1], color=CLASS_COLORS[c],
                       edgecolor="white", linewidth=0.5, s=45, zorder=2, label=f"Class {c}")

    for i in nbr_idx:
        ax.plot([qx, train_X[i, 0]], [qy, train_X[i, 1]], color="#888888", linewidth=0.9,
                 linestyle="-", alpha=0.7, zorder=3)

    radius = nbr_dists.max()
    circle = plt.Circle((qx, qy), radius, fill=False, edgecolor="#c97ef7", linestyle="--",
                          linewidth=1.5, zorder=3)
    ax.add_patch(circle)

    ax.scatter([qx], [qy], color="black", marker="*", s=280, zorder=5, edgecolor="white",
               linewidth=0.8, label="query point")

    ax.set_xlim(PLOT_RANGE)
    ax.set_ylim(PLOT_RANGE)
    ax.set_xlabel("X₁")
    ax.set_ylabel("X₂")
    ax.set_title(f"{mode} with K = {k}")
    if not is_regression:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "The dashed circle reaches exactly as far as the **K-th closest neighbor**, everything "
        "inside it is being used for this prediction. Try changing K in the sidebar and watch "
        "the shaded regions and circle size change together."
    )

    st.info("Do not forget to scale your data before using KNN.", icon="⚠️")

with right_col:
    st.subheader("Prediction for your query point")

    if is_regression:
        st.metric("Predicted value", f"{pred:.2f}")
    else:
        st.metric("Predicted class", f"Class {pred}")

    st.write(
        f"Looking at the **{k} closest** training points to "
        f"(X₁ = {qx:.1f}, X₂ = {qy:.1f})..."
    )

    order = np.argsort(nbr_dists)
    neighbor_rows = []
    for rank, j in enumerate(order, start=1):
        i = nbr_idx[j]
        row = {
            "Rank": rank,
            "Distance": round(float(nbr_dists[j]), 2),
            "X₁": round(float(train_X[i, 0]), 2),
            "X₂": round(float(train_X[i, 1]), 2),
        }
        if is_regression:
            row["Value"] = round(float(train_y[i]), 2)
        else:
            row["Class"] = int(train_y[i])
        neighbor_rows.append(row)

    st.dataframe(neighbor_rows, height=220, use_container_width=True, hide_index=True)

    csv_buffer = StringIO()
    csv_writer = csv.DictWriter(csv_buffer, fieldnames=neighbor_rows[0].keys())
    csv_writer.writeheader()
    csv_writer.writerows(neighbor_rows)
    st.download_button(
        "Download neighbor table as CSV", csv_buffer.getvalue(), "knn_neighbors.csv", "text/csv",
        icon=":material/download:", use_container_width=True,
    )

    st.divider()

    if is_regression:
        st.write(f"**Average of these {k} values → {pred:.2f}**")
        st.latex(r"\hat{y} = \frac{1}{K}\sum_{j=1}^{K} y_j")
    else:
        st.write("**Votes among these neighbors:**")
        vote_counts = Counter(nbr_labels.tolist())
        fig2, ax2 = plt.subplots(figsize=(4, 2.2))
        classes_present = sorted(vote_counts.keys())
        counts = [vote_counts[c] for c in classes_present]
        bar_colors = [CLASS_COLORS[c] for c in classes_present]
        ax2.bar([f"Class {c}" for c in classes_present], counts, color=bar_colors, edgecolor="white")
        ax2.set_ylabel("Votes")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.latex(r"\hat{y} = \text{mode}(y_1, y_2, \dots, y_K)")
        if len(vote_counts) > 1 and vote_counts.most_common(1)[0][1] == vote_counts.most_common(2)[-1][1]:
            st.caption(
                "Note: this vote is tied. This implementation, like scikit-learn's default "
                "behavior, chooses the lowest class index. Other tie-breakers may choose the "
                "closest tied neighbor, use distance-weighted votes, or choose randomly. "
                "With equally distant neighbors, scikit-learn warns that training-data order "
                "can also affect which neighbors are selected."
            )

st.divider()

st.subheader("The idea, in one formula")
st.latex(r"d(\mathbf{a}, \mathbf{b}) = \sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2}")
st.write(
    "Find the distance of query point from every training point using the above mentioned (or some other) formula. "
    "Based on distance, find **K closest points/nearest neighbors**. "
    "Then just do a **majority vote for classification**, or an **average for regression**. "
    "That's all the math KNN needs. "
)

st.info(
    f"**A note on K:** small values of K (like 1 or 2) follow individual nearby points very "
    f"closely and a single unusual point can swing the prediction. "
    f"Larger K smooths things out, but too large and the regions stop reflecting "
    f"real local structure at all. Try changing K from 1 up to 25 and watch this trade-off "
    f"play out in the shaded regions above. In practice, try several K values on validation "
    f"data (or use cross-validation) and choose the K with the best model performance.", icon="⚖️"
)