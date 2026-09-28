"""Two charts, built as deck assets rather than as screen decoration.

Both read the same cached verdicts every other script reads, so a chart can
never drift from the number it was drawn from. Sized for a projector: large
type, values printed on the data, nothing competing with it.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No window: the script writes files and exits.
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from cascade import JUDGE_COST_PER_1000, above, results_for, route
from run_eval import DETECTORS
from src.halueval import build_splits
from src.harness import run
from src.metrics import compute

# Kept in the repository, unlike docs/: these travel into the deck and the
# README, so they cannot live only on one laptop.
CHARTS = Path("charts")

# The interface's own greys, so the slides and the product look like one thing.
INK = "#3f3f46"
MUTED = "#9ca3af"
WARN = "#b91c1c"

# ---- Chart 1: what accuracy hides --------------------------------------------

def methods_chart(cases, results_by_label) -> None:
    """Accuracy and recall side by side, one pair of bars per method.

    Two series because they answer different questions, and the second is the
    one the product lives on: accuracy says how often a method is right at all,
    recall says what share of the real errors it caught. Entailment looks
    respectable on the first and poor on the second, and only the pair shows it.
    """
    labels = list(results_by_label)
    metrics = [compute(cases, r) for r in results_by_label.values()]
    positions = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    ax.bar([p - width / 2 for p in positions], [m.accuracy for m in metrics],
           width, label="Accuracy — how often it is right", color=INK)
    ax.bar([p + width / 2 for p in positions], [m.recall for m in metrics],
           width, label="Recall — the share of real errors it caught", color=MUTED)

    # The line that gives the first pair its meaning: under it, a method is
    # losing to a coin.
    ax.axhline(0.5, color=WARN, linewidth=1, linestyle="--")
    ax.text(-0.45, 0.52, "a coin toss", color=WARN, fontsize=13)

    # Printed on the bars so the chart reads from the back of a room without
    # anyone tracing a value across to an axis.
    for position, m in zip(positions, metrics):
        ax.text(position - width / 2, m.accuracy + 0.02, f"{m.accuracy:.0%}",
                ha="center", fontsize=17, color=INK)
        ax.text(position + width / 2, m.recall + 0.02, f"{m.recall:.0%}",
                ha="center", fontsize=17, color=MUTED)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=17)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])                      # The values are on the bars already.
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="upper left")
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(CHARTS / "methods.png")
    plt.close(fig)


# ---- Chart 2: what the saving costs ------------------------------------------

def tradeoff_chart(cases, cheap_results, judge_results) -> None:
    """Accuracy against the bill, one point per routing rule.

    The shape is the argument: flat at the top — the first cases taken off the
    judge cost nothing — then a steady fall once the cheap stage starts
    deciding cases it has no signal on.
    """
    costs, accuracies = [], []
    for step in range(0, 11):
        chosen, share = route(cases, cheap_results, judge_results, above(step / 10))
        costs.append(share * JUDGE_COST_PER_1000)
        accuracies.append(compute(cases, chosen).accuracy)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.plot(costs, accuracies, color=MUTED, linewidth=2, marker="o",
            markersize=7, markerfacecolor=INK, markeredgecolor=INK)

    # The recommendation, circled rather than described. Index 2 is the rule
    # "escalate above 0.2" — the last point on the flat run, so the last place
    # where paying more still buys something.
    ax.scatter([costs[2]], [accuracies[2]], s=260, facecolor="none",
               edgecolor=WARN, linewidth=2.5, zorder=3)
    ax.annotate("Recommended\nSame accuracy, 18% cheaper",
                xy=(costs[2], accuracies[2]),
                xytext=(costs[2] - 0.010, accuracies[2] + 0.004),
                fontsize=15, color=WARN, ha="right", va="center")

    # Both ends are named as what they are — the floor and the ceiling this
    # curve runs between — rather than as anything being recommended. The index
    # is the step it came from: 10 escalates nothing, 0 escalates everything.
    ax.annotate("No judge at all",
                xy=(costs[10], accuracies[10]),
                xytext=(costs[10] + 0.012, accuracies[10]),
                fontsize=13, color=INK, va="center")

    ax.annotate("Judge on\nevery answer",
                xy=(costs[0], accuracies[0]),
                xytext=(costs[0], accuracies[0] - 0.008),
                fontsize=13, color=INK, ha="left", va="top")

    # Room at both edges for those labels, which otherwise run off the plot.
    ax.set_xlim(-0.015, 0.36)

    # Headroom above the flat run, so the label beside the circled point is not
    # struck through by the axis line above it.
    ax.set_ylim(0.625, 0.85)

    ax.set_xlabel("Cost per 1,000 checks, US dollars", fontsize=15,
                  labelpad=14)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.spines[["right"]].set_visible(False)

    # What the rule actually varies is the share of answers the judge reads;
    # the bill is only its consequence, and the two are exactly proportional,
    # so a second axis states both without distorting either.
    share_axis = ax.secondary_xaxis(
        "top",
        functions=(lambda cost: cost / JUDGE_COST_PER_1000 * 100,
                   lambda share: share / 100 * JUDGE_COST_PER_1000))
    share_axis.set_xlabel("Share of answers sent to the AI judge",
                          fontsize=15, labelpad=14)
    share_axis.tick_params(labelsize=13)
    share_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0f}%"))

    # Ticks named rather than left to the axis: the plot runs a little past the
    # rightmost point to make room for its label, and an automatic axis would
    # print a 120% that cannot exist.
    share_axis.set_xticks([0, 20, 40, 60, 80, 100])

    # The sentence the chart cannot draw: what a single point on it is.
    fig.text(0.5, 0.015,
             "Each point is one rule: the cheap check reads every answer, "
             "and only those it scores above a cut-off go to the judge.",
             ha="center", fontsize=13, color="#71717a")

    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(CHARTS / "tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    CHARTS.mkdir(exist_ok=True)
    _, cases = build_splits()

    # The embedding detector is picked by threshold as well as by name: the
    # list holds two of them, and the first is the degenerate 0.00 tuning that
    # calls everything grounded.
    similarity = next(d for d in DETECTORS
                      if d.name == "embeddings" and d.threshold == 0.5)

    results = {
        "Text similarity": run(similarity, cases, token_budget=0),
        "Entailment": results_for("entailment", cases),
        "AI judge": results_for("llm_judge", cases),
    }

    methods_chart(cases, results)
    tradeoff_chart(cases, results["Entailment"], results["AI judge"])
    print(f"written: {CHARTS / 'methods.png'}, {CHARTS / 'tradeoff.png'}")

