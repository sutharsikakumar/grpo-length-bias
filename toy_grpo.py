"""Toy GRPO experiment: does per-response length normalization let wrong answers grow longer?

Variant A: original GRPO (advantage divided by group std; loss averaged over each response's own length)
Variant B: Dr. GRPO-style (no std division; loss summed over tokens and divided by a fixed constant)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN columns are expected late in training

# ---------- Setup ----------
MAX_LEN = 10                  # longest possible response
GOOD, BAD, STOP = 0, 1, 2     # the three actions
G = 36                        # group size (responses sampled per step)
SEEDS = [0, 1, 2, 3, 4]


def new_policy():
    # One row of logits per position; all zeros means every action starts at probability 1/3.
    return torch.zeros(MAX_LEN, 3, requires_grad=True)


def sample_response(policy):
    # Write a response one action at a time, recording each action's log-probability.
    tokens, logprobs = [], []
    for t in range(MAX_LEN):
        probs = torch.softmax(policy[t], dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprobs.append(dist.log_prob(action))
        if action.item() == STOP:
            break
        tokens.append(action.item())
    return tokens, torch.stack(logprobs)


def reward(tokens):
    # Correct = at least one token and no bad tokens.
    return 1.0 if len(tokens) > 0 and BAD not in tokens else 0.0


# ---------- One GRPO training step ----------
def grpo_step(policy, optimizer, variant):
    samples = [sample_response(policy) for _ in range(G)]
    rewards = torch.tensor([reward(tok) for tok, _ in samples])

    if variant == "A":  # original GRPO: std-normalized advantage, per-response length average
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        losses = [-(a * lp).mean() for a, (_, lp) in zip(adv, samples)]
        loss = torch.stack(losses).mean()
    else:  # "B", Dr. GRPO-style: raw advantage, sum over tokens, fixed normalizer
        adv = rewards - rewards.mean()
        losses = [-(a * lp).sum() for a, (_, lp) in zip(adv, samples)]
        loss = torch.stack(losses).sum() / (G * (MAX_LEN + 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return samples, rewards


# ---------- Evaluation (no training) ----------
@torch.no_grad()
def evaluate(policy, n=500):
    lens_wrong, lens_correct = [], []
    for _ in range(n):
        tok, _ = sample_response(policy)
        (lens_correct if reward(tok) == 1 else lens_wrong).append(len(tok))
    acc = len(lens_correct) / n
    mean = lambda x: sum(x) / len(x) if x else float("nan")
    return acc, mean(lens_wrong), mean(lens_correct), len(lens_wrong)


# ---------- Training loop ----------
def train(variant, steps=300, seed=0, eval_every=10):
    torch.manual_seed(seed)
    policy = new_policy()
    opt = torch.optim.Adam([policy], lr=0.05)
    log = {"step": [], "acc": [], "len_wrong": [], "len_correct": [], "n_wrong": []}
    for step in range(steps + 1):
        if step % eval_every == 0:
            acc, lw, lc, nw = evaluate(policy)
            for k, v in zip(log, [step, acc, lw, lc, nw]):
                log[k].append(v)
        if step < steps:
            grpo_step(policy, opt, variant)
    return log


# ---------- Figure style (paper-like: serif text, open axes, faint grid, markers) ----------
BLUE = "#1B3A8C"    # GRPO
GREEN = "#2E8B57"   # Dr. GRPO-style
STYLE = {
    "A": dict(color=BLUE, marker="*", markersize=9, label="GRPO (per-response norm)"),
    "B": dict(color=GREEN, marker="D", markersize=6, label="Dr. GRPO-style (constant norm)"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,  # draw gridlines behind bars and lines
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "savefig.bbox": "tight",
})


def summarize(results, v, key, min_wrong=20):
    arr = np.array([r[key] for r in results[v]], dtype=float)
    if key == "len_wrong":  # hide points with too few wrong answers to measure reliably
        nw = np.array([r["n_wrong"] for r in results[v]])
        arr[nw < min_wrong] = np.nan
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr


# ---------- Experiment and plot ----------
if __name__ == "__main__":
    results = {v: [train(v, seed=s) for s in SEEDS] for v in ["A", "B"]}
    steps = np.array(results["A"][0]["step"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1.0, 0.8]}
    )

    # Panel 1: wrong-answer length over training (mean +/- std across seeds)
    for v in ["A", "B"]:
        m, s, _ = summarize(results, v, "len_wrong")
        st = STYLE[v]
        ax1.plot(steps, m, color=st["color"], marker=st["marker"], markersize=st["markersize"],
                 linewidth=2, label=st["label"])
        ax1.fill_between(steps, m - s, m + s, color=st["color"], alpha=0.15, linewidth=0)
        last = np.where(~np.isnan(m))[0][-1]
        ax1.annotate(f"{m[last]:.2f}", (steps[last], m[last]), textcoords="offset points",
                     xytext=(6, 4), color=st["color"], fontsize=10)
    ax1.set(xlabel="Training step", ylabel="Mean length of wrong answers (tokens)")
    ax1.set_title(f"Wrong answers grow under GRPO  (G={G}, {len(SEEDS)} seeds)", fontweight="bold")
    ax1.legend(loc="lower left")

    # Panel 2: accuracy over training (both learn equally fast)
    for v in ["A", "B"]:
        m, s, _ = summarize(results, v, "acc")
        st = STYLE[v]
        ax2.plot(steps, m * 100, color=st["color"], linewidth=2, label=st["label"],
                 linestyle="-" if v == "A" else "--")
        ax2.fill_between(steps, (m - s) * 100, (m + s) * 100, color=st["color"], alpha=0.12, linewidth=0)
    ax2.set(xlabel="Training step", ylabel="Accuracy (%)", ylim=(0, 102))
    ax2.set_title("Same accuracy", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)

    # Panel 3: bar chart of wrong-answer length at a fixed step where both are measurable
    m_a, _, arr_a = summarize(results, "A", "len_wrong")
    m_b, _, arr_b = summarize(results, "B", "len_wrong")
    both = np.where(~np.isnan(m_a) & ~np.isnan(m_b))[0]
    idx = both[-1]  # latest step where both variants still have enough wrong answers
    vals = [np.nanmean(arr_a[:, idx]), np.nanmean(arr_b[:, idx])]
    errs = [np.nanstd(arr_a[:, idx]), np.nanstd(arr_b[:, idx])]
    bars = ax3.bar([0, 1], vals, yerr=errs, capsize=5, width=0.6,
                   color=[BLUE, GREEN], edgecolor=[BLUE, GREEN], error_kw={"ecolor": "#333333"})
    for b, val, err in zip(bars, vals, errs):  # label sits above the error bar
        ax3.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, val + err), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=11)
    ax3.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.18)
    ax3.set_xticks([0, 1], ["GRPO", "Dr. GRPO"])
    ax3.set(ylabel="Wrong-answer length (tokens)")
    ax3.set_title(f"At step {steps[idx]}", fontweight="bold")
    ax3.grid(axis="x", visible=False)

    plt.tight_layout()
    out = f"length_bias_G{G}.png"
    plt.savefig(out, dpi=200)
    plt.savefig(out.replace(".png", ".pdf"))  # vector version for the blog post
    print(f"Saved {out} and PDF")
   