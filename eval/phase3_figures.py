from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EVAL = Path("results/eval")
OUT = Path("results/figures")

# set name -> (method, lambda).  lambda = 1 + w for CFG, so both share one axis.
SETS = {
    "cg_AcyP_lambda0": ("CG", 0.0), "cg_AcyP_lambda0.5": ("CG", 0.5),
    "cg_AcyP_lambda1": ("CG", 1.0), "cg_AcyP_lambda2": ("CG", 2.0),
    "cg_AcyP_lambda4": ("CG", 4.0),
    "cfg_null": ("CFG", 0.0), "cfg_AcyP_w0": ("CFG", 1.0), "cfg_AcyP_w0.5": ("CFG", 1.5),
    "cfg_AcyP_w1": ("CFG", 2.0), "cfg_AcyP_w2": ("CFG", 3.0), "cfg_AcyP_w3": ("CFG", 4.0),
}
CONTROLS = {"control_real": "-", "control_shuffled": "--"}


def load():
    rows = []
    for name in list(SETS) + list(CONTROLS):
        path = EVAL / name / "summary.csv"
        if not path.exists():
            print(f"missing {path}, skipping")
            continue
        row = pd.read_csv(path).iloc[0].to_dict()
        row["set"] = name
        row["method"], row["lam"] = SETS.get(name, ("control", float("nan")))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("set")

    for predictor in ("guide", "judge"):
        path = EVAL / f"family_membership_{predictor}.csv"
        if path.exists():
            table[f"membership_{predictor}"] = pd.read_csv(path).set_index("set")["frac_target"]
    return table


def controls(ax, table, column):
    for name, style in CONTROLS.items():
        if name in table.index and column in table:
            ax.axhline(table.loc[name, column], ls=style, c="grey", lw=1)
            ax.text(0.01, table.loc[name, column], name, fontsize=7, c="grey",
                    va="bottom", transform=ax.get_yaxis_transform())


def tradeoff(table):
    if "mean_plddt" not in table:
        print("no mean_plddt yet (run eval with --fold), skipping tradeoff.png")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, marker in (("CG", "o"), ("CFG", "s")):
        d = table[table["method"] == method].sort_values("lam")
        ax.plot(d["membership_guide"], d["mean_plddt"], marker=marker, label=method)
        for _, r in d.iterrows():
            ax.annotate(f"λ={r['lam']:g}", (r["membership_guide"], r["mean_plddt"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
    controls(ax, table, "mean_plddt")
    ax.set_xlabel("predicted target-family membership")
    ax.set_ylabel("mean pLDDT")
    ax.set_title("Guidance trade-off (AcyP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=200)


def collapse(table):
    columns = ["frac_valid", "frac_unique", "nearest_identity_median"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, column in zip(axes, columns):
        for method, marker in (("CG", "o"), ("CFG", "s")):
            d = table[table["method"] == method].sort_values("lam")
            ax.plot(d["lam"], d[column], marker=marker, label=method)
        controls(ax, table, column)
        ax.set_xlabel("guidance strength λ")
        ax.set_title(column)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "collapse.png", dpi=200)


def circularity(table):
    if "membership_judge" not in table:
        print("no judge scores yet, skipping circularity.png")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, marker in (("CG", "o"), ("CFG", "s")):
        d = table[table["method"] == method].sort_values("lam")
        ax.plot(d["lam"], d["membership_guide"], marker=marker, label=f"{method} - guiding")
        ax.plot(d["lam"], d["membership_judge"], marker=marker, ls="--",
                label=f"{method} - independent judge")
    ax.set_xlabel("guidance strength λ")
    ax.set_ylabel("predicted target-family membership")
    ax.set_title("Scored by the classifier that guided them, vs an independent one")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "circularity.png", dpi=200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = load()
    table.to_csv(OUT / "phase3_table.csv")
    print(table.to_string(), "\n")
    tradeoff(table)
    collapse(table)
    circularity(table)
    print("Figures in", OUT)


if __name__ == "__main__":
    main()