"""Figures for the analysis report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REGIME_COLORS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a"]


def fan_chart(paths: np.ndarray, title: str, path: str, n_show: int = 150, seed: int = 0):
    rng = np.random.default_rng(seed)
    steps = np.arange(paths.shape[1])
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i in rng.choice(paths.shape[0], min(n_show, paths.shape[0]), replace=False):
        ax.plot(steps, paths[i], lw=0.6, alpha=0.35)
    bands = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(steps, bands[0], bands[4], color="black", alpha=0.08, label="5–95% band")
    ax.fill_between(steps, bands[1], bands[3], color="black", alpha=0.12, label="25–75% band")
    ax.plot(steps, bands[2], color="black", lw=2, label="Median")
    ax.set(title=title, xlabel="Trading days ahead", ylabel="Price")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def terminal_distributions(path_sets: dict[str, np.ndarray], title: str, path: str, alpha: float = 0.05):
    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, p), color in zip(path_sets.items(), ["#264653", "#e76f51", "#2a9d8f", "#8d99ae"]):
        rets = p[:, -1] / p[:, 0] - 1
        ax.hist(rets * 100, bins=120, density=True, histtype="step", lw=1.8, color=color, label=name)
        ax.axvline(np.quantile(rets, alpha) * 100, color=color, ls="--", lw=1)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(title=title + f"  (dashed: {int(alpha * 100)}% quantile)", xlabel="Horizon return (%)", ylabel="Density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def regime_history(prices: pd.Series, probs: np.ndarray, title: str, path: str):
    """Price coloured by most likely filtered regime, plus stacked filtered probabilities."""
    idx = prices.index[1:]
    px = prices.iloc[1:].to_numpy()
    state = probs.argmax(axis=1)
    k = probs.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(idx, px, lw=0.8, color="#495057", alpha=0.6)
    for s in range(k):
        m = state == s
        ax1.scatter(idx[m], px[m], s=4, color=_REGIME_COLORS[s % 5], label=f"State {s}", zorder=3)
    ax1.set(title=title, ylabel="Price")
    ax1.legend(loc="upper left", title="0 = lowest vol")
    ax1.grid(alpha=0.3)
    ax2.stackplot(idx, probs.T, colors=[_REGIME_COLORS[s % 5] for s in range(k)], alpha=0.85)
    ax2.set(ylabel="Filtered P(state)", ylim=(0, 1))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def var_backtest(results: dict[str, pd.DataFrame], alpha: float, horizon: int, path: str):
    fig, ax = plt.subplots(figsize=(13, 6))
    first = next(iter(results.values()))
    ax.bar(first.index, first["realized"] * 100, width=15, color="#adb5bd", label=f"Realised {horizon}-bar return")
    for (name, bt), color in zip(results.items(), ["#264653", "#e76f51"]):
        ax.plot(bt.index, -bt["var"] * 100, color=color, lw=1.6, label=f"{name} VaR {int((1 - alpha) * 100)}%")
        br = bt[bt["breach"]]
        ax.scatter(br.index, br["realized"] * 100, color=color, s=28, zorder=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(title="Out-of-sample VaR backtest (dots = breaches)", ylabel="Return (%)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
