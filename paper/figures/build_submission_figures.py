"""Build development-only publication figures from the dashboard's verified records.

Requires an already installed matplotlib for authoring; the dashboard needs none.
No model, dataset manifest, held-out image, or training entry point is opened.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard.backend.artifacts import ArtifactStore


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import FancyArrowPatch

    out = ROOT / "paper/figures"
    store = ArtifactStore()
    workspace = store.workspace()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    generated = []

    def save(fig, name):
        fig.savefig(out / (name + ".svg"), bbox_inches="tight", metadata={"Date": None})
        svg = out / (name + ".svg")
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                       encoding="utf-8")
        fig.savefig(out / (name + ".png"), bbox_inches="tight", dpi=180,
                    metadata={"Software": "CrossModalLab stored-evidence figure builder"})
        generated.extend([name + ".svg", name + ".png"])
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    examples = [("Entailed: Yes", [(0, 1), (1, 2)], "A < B, B < C implies A < C"),
                ("Contradicted: No", [(1, 0), (2, 1)], "C < B < A rules out A < C"),
                ("Underdetermined", [(0, 1), (2, 1)], "A < C < B or C < A < B")]
    for ax, (title, edges, caption) in zip(axes, examples):
        ax.set(xlim=(-.5, 2.5), ylim=(-.7, .8), title=title)
        ax.axis("off")
        for index, letter in enumerate("ABC"):
            ax.text(index, 0, letter, ha="center", va="center", fontsize=15,
                    bbox={"boxstyle": "circle,pad=.45", "fc": "#eff5ee", "ec": "#267864"})
        for start, end in edges:
            ax.add_patch(FancyArrowPatch((start, .08), (end, .08),
                         shrinkA=24, shrinkB=24, arrowstyle="-|>",
                         mutation_scale=14, color="#267864", linewidth=2))
        ax.text(1, -.48, caption, ha="center", fontsize=10)
    fig.suptitle("Strict left-of: three evidence states (logical schematic)", y=1.03)
    save(fig, "01_evidence_states")

    fig, ax = plt.subplots(figsize=(10.5, 4))
    ax.axis("off")
    stages = [("Controlled image + text", .02, .72), ("Stored VLM outputs", .36, .72),
              ("Behavioral / format audit", .70, .72), ("Symbolic ground truth", .02, .22),
              ("Probes / intervention controls", .36, .22), ("Dashboard + evidence registry", .70, .22)]
    for text, x, y in stages:
        ax.text(x, y, text, transform=ax.transAxes, fontsize=10,
                bbox={"boxstyle": "round,pad=.8", "fc": "#eff5ee", "ec": "#267864"})
    for a, b in [((.26,.74),(.35,.74)), ((.60,.74),(.69,.74)),
                 ((.12,.65),(.12,.31)), ((.24,.32),(.70,.64)),
                 ((.49,.65),(.49,.31)), ((.82,.65),(.82,.31)), ((.63,.24),(.69,.24))]:
        ax.annotate("", xy=b, xytext=a, xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": "#267864"})
    ax.set_title("CrossModalLab research pipeline (conceptual)", pad=18)
    ax.text(.5, -.06, "External probes are correlational; surviving controls establish no relation-specific circuit.",
            ha="center", transform=ax.transAxes, fontsize=10)
    save(fig, "02_research_pipeline")

    ids = ["frozen", "ce5", "es5", "ce50"]
    conditions = {c["id"]: c for c in workspace["conditions"]}
    names = ["Frozen", "CE / 5 steps", "CE+ES / 5 steps", "CE / 50 steps"]
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    x = list(range(4))
    bars = ax.bar([v-.18 for v in x], [conditions[k]["metrics"]["balanced_accuracy"] for k in ids],
                  width=.36, label="Historical semantic balanced accuracy", color="#267864")
    ax.bar([v+.18 for v in x], [conditions[k]["metrics"]["exact_compliance"] for k in ids],
           width=.36, label="Exact-label compliance", color="#c29a55")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                f"{bar.get_height():.3f}", ha="center", fontsize=10)
    ax.set(ylim=(0, 1.32), xticks=x, xticklabels=names, ylabel="Fraction",
           title="Recorded Qwen development checkpoints: 18 cases / 6 families")
    ax.set_yticks([i/5 for i in range(6)])
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    fig.text(.08, -.01, "Development reuse; unequal training budgets. Prefix parser accepts malformed CE suffixes.\n"
             "No superiority claim; descriptive family intervals are in development_summary.json.", fontsize=9)
    save(fig, "03_checkpoint_comparison")

    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    for j, (label, color) in enumerate(zip(("Yes", "No", "Undetermined"), ("#267864", "#6d82a4", "#bd9556"))):
        vals = [conditions[k]["metrics"]["per_class_recall"][label] for k in ids]
        ax.bar([v+(j-1)*.24 for v in x], vals, width=.24, label=label, color=color)
    ax.set(ylim=(0, 1.2), xticks=x, xticklabels=names, ylabel="Recall (six cases per class)",
           title="Class-specific development recall")
    ax.set_yticks([i/5 for i in range(6)])
    ax.legend(ncol=3, frameon=False, loc="upper center", fontsize=10)
    fig.text(.09, -.01, "Same 18 stored validation cases; one seed; repeatedly inspected development data.\n"
             "CE+ES has only five steps. These are descriptive observations, not generalization estimates.", fontsize=9)
    save(fig, "04_class_recall")

    case = next(c for c in workspace["cases"] if c["sample_id"] == "xc_ho_0011::underdetermined")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"width_ratios": [1, 1.45]})
    axes[0].imshow(mpimg.imread(store.image(case["sample_id"])))
    axes[0].axis("off")
    axes[0].set_title("Original recorded input image", fontsize=11)
    axes[1].axis("off")
    lines = [case["sample_id"], "", "A: golden disc     B: teal wedge     C: crimson block",
             "Image: A < B.     Text: C < B.     Query: A < C?", "",
             "Compatible orders: A < C < B and C < A < B.", "Symbolic answer: Undetermined.", ""]
    for key, name in [("frozen", "Frozen Qwen"), ("es5", "CE+ES step 5")]:
        raw = case["predictions"][key]["raw_completion"]
        lines.append(f"{name}: raw {json.dumps(raw)} -> unsupported No")
    axes[1].text(0, .98, "\n".join(lines), va="top", fontsize=10, linespacing=1.6)
    fig.suptitle("Observed behavioral failure; internal cause remains unverified", fontsize=12)
    save(fig, "05_recorded_failure")

    fields = ["condition", "steps", "prompt", "n", "families", "balanced_accuracy", "macro_f1",
              "recall_yes", "recall_no", "recall_undetermined", "invalid_output_rate", "exact_compliance"]
    with (ROOT / "paper/tables/submission_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in workspace["conditions"]:
            m = c["metrics"]
            writer.writerow({"condition": c["id"], "steps": c["step"], "prompt": c["prompt"],
                **{k: m[k] for k in fields if k in m},
                **{"recall_"+l.lower(): m["per_class_recall"][l] for l in ("Yes", "No", "Undetermined")}})
    demo_cases = [{"sample_id": c["sample_id"], "prompt": c["prompt"],
                   "premises": c["premises"], "query": c["query"],
                   "ground_truth_label": c["ground_truth_label"], "image_sha256": c["image_sha256"],
                   "orderings": c["reasoning"]["compatible_orderings"],
                   "predictions": {k: c["predictions"][k] for k in ids}}
                  for c in workspace["cases"] if c["family_id"] == "xc_ho_0011"]
    target = ROOT / "submission/2026-10-28"
    target.mkdir(parents=True, exist_ok=True)
    (target / "demo_examples.json").write_text(json.dumps(demo_cases, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    manifest = {"classification": "PRELIMINARY DEVELOPMENT; schematic figures explicitly conceptual",
                "holdout_used": False, "new_inference": False, "sources": workspace["sources"],
                "files": {f: hashlib.sha256((out/f).read_bytes()).hexdigest() for f in generated}}
    (out / "submission_figure_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"figures": 5, "files": len(generated), "cases": len(demo_cases), "new_inference": False}))


if __name__ == "__main__":
    main()
