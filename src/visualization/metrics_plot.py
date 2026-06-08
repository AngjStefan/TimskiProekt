import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_regression_results(
    targets: np.ndarray,
    preds: np.ndarray,
    model_name: str = "Model",
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # scatter
    axes[0].scatter(targets, preds, alpha=0.5, s=10)
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=1)
    axes[0].set_xlabel("True EF")
    axes[0].set_ylabel("Predicted EF")
    axes[0].set_title(f"{model_name} — Predicted vs True")
    axes[0].axis("equal")

    # error histogram
    errors = preds - targets
    axes[1].hist(errors, bins=50, alpha=0.7)
    axes[1].axvline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Prediction Error")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"MAE = {np.abs(errors).mean():.4f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_segmentation_results(
    dice_scores: np.ndarray,
    ious: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(dice_scores, bins=50, alpha=0.7)
    axes[0].axvline(dice_scores.mean(), color="r", linestyle="--",
                    label=f"Mean={dice_scores.mean():.4f}")
    axes[0].set_xlabel("Dice Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Dice Score Distribution")
    axes[0].legend()

    axes[1].hist(ious, bins=50, alpha=0.7)
    axes[1].axvline(ious.mean(), color="r", linestyle="--",
                    label=f"Mean={ious.mean():.4f}")
    axes[1].set_xlabel("IoU")
    axes[1].set_ylabel("Count")
    axes[1].set_title("IoU Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def create_comparison_bar_chart(results: dict) -> go.Figure:
    """Plotly interactive bar chart comparing models."""
    models = list(results.keys())
    metrics = list(results[models[0]].keys())
    n_metrics = len(metrics)

    fig = make_subplots(rows=1, cols=n_metrics,
                        subplot_titles=metrics,
                        shared_yaxes=False)

    colors = ["#636EFA", "#EF553B", "#00CC96"]

    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        fig.add_trace(
            go.Bar(name=metric, x=models, y=vals,
                   marker_color=colors[:len(models)],
                   text=[f"{v:.4f}" for v in vals],
                   textposition="outside"),
            row=1, col=i + 1,
        )

    fig.update_layout(
        title_text="Model Comparison",
        showlegend=False,
        height=500,
        width=400 * n_metrics,
    )
    return fig
