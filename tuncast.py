"""Shortcuts for the project.

    python tuncast.py app                     open the web app
    python tuncast.py run [--from N]          run the pipeline (steps N-7)
    python tuncast.py test                    run the tests
    python tuncast.py paper                   rebuild paper/main.pdf
    python tuncast.py stations [--model IFS]  day-1 Tmax error per station, raw vs corrected
    python tuncast.py day Sfax 2026-08-23     the seven forecasts for one day
"""
import argparse
import subprocess
import sys

import pandas as pd

from pipeline.config import MODELS, RESULTS, ROOT, STATIONS, TEST_END, TEST_START
from pipeline.utils import mae

IDS = {name.lower(): wmo for wmo, (name, *_rest) in STATIONS.items()}


def python(*args):
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def predictions():
    path = RESULTS / "predictions_temperature.csv"
    if not path.exists():
        sys.exit("No predictions yet. Run:  python tuncast.py run")
    t = pd.read_csv(path, dtype={"station": str}, parse_dates=["date"])
    return t[(t.date >= TEST_START) & (t.date <= TEST_END)]


def bar(value, scale, width=24):
    return "█" * round(min(value / scale, 1) * width)


def stations(model):
    t = predictions()
    t = t[(t.model == model) & (t.lead == 1) & (t.target == "tmax")].dropna(subset=["obs", "raw", "lightgbm"])
    rows = [(STATIONS[w][0], mae(g.obs, g.raw), mae(g.obs, g.lightgbm)) for w, g in t.groupby("station")]
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    scale = max(r[1] for r in rows)
    print(f"\n{model}, Tmax, 1 day ahead, mean absolute error (°C)\n")
    print(f"{'station':<10} {'raw':>5} {'fixed':>6}")
    for name, raw, fixed in rows:
        print(f"{name:<10} {raw:5.2f} {fixed:6.2f}  {bar(fixed, scale)}\033[2m{bar(raw - fixed, scale)}\033[0m")
    print("\nsolid = error left after correction, faint = error removed\n")


def day(station, date, model):
    wmo = IDS.get(station.lower())
    if wmo is None:
        sys.exit(f"Unknown station. Choose from: {', '.join(n for n, *_ in STATIONS.values())}")
    t = predictions()
    t = t[(t.model == model) & (t.station == wmo) & (t.date == pd.Timestamp(date))]
    if t.empty:
        sys.exit(f"No {model} forecasts for {station} on {date} (test period {TEST_START} to {TEST_END}).")
    print(f"\n{STATIONS[wmo][0]}, {date}, {model}\n")
    for target in ("tmax", "tmin"):
        g = t[t.target == target].sort_values("lead", ascending=False)
        observed = g.obs.dropna()
        print(f"{target.capitalize()}  observed {observed.iloc[0]:.1f} °C" if len(observed) else target.capitalize())
        print(f"  {'issued':<12} {'raw':>6} {'fixed':>6}  {'80% range':>13}")
        for _, r in g.iterrows():
            interval = (f"{r.conformal_low:5.1f}-{r.conformal_high:4.1f}"
                        if pd.notna(r.conformal_low) else "")
            print(f"  {int(r.lead)} day{'s' if r.lead > 1 else ' '} before {r.raw:6.1f} {r.lightgbm:6.1f}  {interval:>13}")
        print()


def main():
    parser = argparse.ArgumentParser(description="TunCast shortcuts")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("app")
    run = sub.add_parser("run")
    run.add_argument("--from", dest="start", type=int, default=1)
    sub.add_parser("test")
    sub.add_parser("paper")
    st = sub.add_parser("stations")
    st.add_argument("--model", default="IFS", choices=list(MODELS))
    d = sub.add_parser("day")
    d.add_argument("station")
    d.add_argument("date")
    d.add_argument("--model", default="IFS", choices=list(MODELS))
    args = parser.parse_args()

    if args.command == "app":
        sys.exit(python("app.py"))
    if args.command == "run":
        sys.exit(python("run_all.py", str(args.start)))
    if args.command == "test":
        sys.exit(python("-m", "pytest", "-q"))
    if args.command == "paper":
        sys.exit(python("paper/make_paper.py"))
    if args.command == "stations":
        stations(args.model)
    if args.command == "day":
        day(args.station, args.date, args.model)


if __name__ == "__main__":
    main()
