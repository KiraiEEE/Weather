"""Cached downloads and scoring helpers."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests

from pipeline.config import RAW


def download(url, params=None, check=None, tries=5):
    """Download a URL once and keep a copy in data/raw/.
    `check(content)` can reject bad answers (some servers send an error message with status 200).
    Returns None if the file does not exist (HTTP 404)."""
    key = hashlib.md5((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = RAW / key
    if path.exists():
        return path.read_bytes()
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, timeout=180)
        except requests.RequestException as e:
            print(f"  network error ({e}), retry {attempt}")
            time.sleep(10 * attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 200 and (check is None or check(r.content)):
            path.write_bytes(r.content)
            return r.content
        print(f"  bad answer (HTTP {r.status_code}: {r.text[:80]!r}), retry {attempt}")
        time.sleep(30 * attempt)
    raise RuntimeError(f"Download failed: {url}\nRun the step again later; finished downloads are kept.")


# scores
def mae(obs, pred):
    return float(np.mean(np.abs(np.asarray(obs) - np.asarray(pred))))


def brier(event, prob):
    return float(np.mean((np.asarray(prob) - np.asarray(event)) ** 2))


def coverage(obs, low, high):
    obs = np.asarray(obs)
    return float(np.mean((obs >= np.asarray(low)) & (obs <= np.asarray(high))))


def bootstrap_ci(dates, diff, reps=1000, block_days=7, seed=0):
    """95% confidence interval of mean(diff), resampling whole weeks
    (days next to each other are correlated, so single days must not be resampled)."""
    df = pd.DataFrame({"week": (pd.to_datetime(np.asarray(dates)) - pd.Timestamp("2000-01-01")).days // block_days,
                       "diff": np.asarray(diff)})
    weeks = df.groupby("week")["diff"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(weeks), size=(reps, len(weeks)))
    means = weeks[picks, 0].sum(axis=1) / weeks[picks, 1].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
