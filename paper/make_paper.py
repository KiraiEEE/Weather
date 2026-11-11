"""Build the short paper: results CSVs -> LaTeX tables -> paper/main.pdf

Run after the pipeline:  .venv/Scripts/python.exe paper/make_paper.py
Needs pdflatex (MiKTeX or TeX Live). Numbers in the tables are never typed by hand.
"""
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
MODELS = ["IFS", "AIFS", "GFS"]


def temperature_table():
    s = pd.read_csv(RESULTS / "temperature_skill.csv")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Target & Lead (d) & Days & MAE raw & MAE LGBM & Gain \% [95\% CI] \\", r"\midrule"]
    for model in MODELS:
        for target in ("tmax", "tmin"):
            for lead in (1, 3, 5, 7):
                r = s[(s.model == model) & (s.target == target) & (s.lead == lead)]
                if r.empty:
                    continue
                r = r.iloc[0]
                lines.append(f"{model} & {target} & {lead} & {int(r.days)} & {r['MAE raw']:.2f} & "
                             f"{r['MAE lightgbm']:.2f} & {r['improvement %']:.1f} "
                             f"[{r['95% CI low %']:.1f}, {r['95% CI high %']:.1f}] \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def baselines_table():
    m = pd.read_csv(RESULTS / "temperature_mae.csv")
    m = m[m.lead.isin([1, 4, 7])]
    cols = ["raw", "climatology", "persistence", "bias_corrected", "linear", "lightgbm"]
    head = ["Raw", "Clim.", "Persist.", "Bias-corr.", "Linear", "LGBM"]
    lines = [r"\begin{tabular}{lll" + "r" * len(cols) + "}", r"\toprule",
             "Model & Target & Lead & " + " & ".join(head) + r" \\", r"\midrule"]
    for model in MODELS:
        for target in ("tmax", "tmin"):
            for _, r in m[(m.model == model) & (m.target == target)].sort_values("lead").iterrows():
                values = [r[c] for c in cols]
                best = min(values)
                cells = [rf"\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}" for v in values]
                lines.append(f"{model} & {target} & {int(r.lead)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def interval_table():
    t = pd.read_csv(RESULTS / "intervals.csv")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{Quantile LGBM} & \multicolumn{2}{c}{+ Conformal} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"Model & Target & Cover \% & Width & Cover \% & Width \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.model} & {r.target} & {r['lightgbm coverage %']:.1f} & {r['lightgbm width']:.2f} & "
                     f"{r['conformal coverage %']:.1f} & {r['conformal width']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rain_table():
    t = pd.read_csv(RESULTS / "rain_scores.csv")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Model & Method & Brier & BSS & ROC AUC & PR AUC \\", r"\midrule"]
    for model in MODELS:
        for _, r in t[t.model == model].iterrows():
            lines.append(f"{model} & {r.method} & {r.Brier:.3f} & {r['Brier skill vs climatology']:.3f} & "
                         f"{r['ROC AUC']:.3f} & {r['PR AUC']:.3f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    tables = {"tab_temperature": temperature_table, "tab_baselines": baselines_table,
              "tab_intervals": interval_table, "tab_rain": rain_table}
    for name, make in tables.items():
        (PAPER / f"{name}.tex").write_text(make(), encoding="utf-8")
    figures = PAPER / "figures"
    figures.mkdir(exist_ok=True)
    for png in (RESULTS / "figures").glob("*.png"):
        shutil.copy(png, figures / png.name)
    for _ in range(2):          # twice, so references are resolved
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=PAPER, check=False,
                       stdout=subprocess.DEVNULL)
    print("Built paper/main.pdf" if (PAPER / "main.pdf").exists() else "pdflatex failed: see paper/main.log")


if __name__ == "__main__":
    main()
