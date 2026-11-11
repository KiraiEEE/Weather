"""GRU vs LightGBM on the IFS temperature error (optional).

Same months, rows and target as step 5; the GRU also gets the last 14 days of observations
up to the issue day. Needs torch: pip install -r requirements-gru.txt (CPU is fine).

Writes results/gru_vs_lightgbm.md
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from pipeline.config import CLEAN, CLIMATOLOGY_END, GAP_DAYS, RESULTS, SEED, TEST_END, TEST_START
from pipeline.step4_features import TEMP_FEATURES
from pipeline.utils import mae

MODEL = "IFS"
DAYS = 14


class GRUModel(nn.Module):
    def __init__(self, n_static):
        super().__init__()
        self.gru = nn.GRU(input_size=3, hidden_size=32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32 + n_static, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, sequence, static):
        _, hidden = self.gru(sequence)
        return self.head(torch.cat([hidden[-1], static], dim=1)).squeeze(1)


def sequences(rows, obs):
    """For each row: tmax, tmin, rain of the 14 days ending on the issue day (rain: day before)."""
    out = np.zeros((len(rows), DAYS, 3), dtype=np.float32)
    for i, (station, issue) in enumerate(zip(rows.station, rows.issue, strict=True)):
        window = obs[station].loc[issue - pd.Timedelta(days=DAYS - 1): issue]
        out[i, DAYS - len(window):] = window.to_numpy()
    return torch.tensor(out)


def train_and_predict(train, test, obs, var):
    torch.manual_seed(SEED)
    mean, std = train[TEMP_FEATURES].mean(), train[TEMP_FEATURES].std().replace(0, 1)
    def static(d):
        return torch.tensor(((d[TEMP_FEATURES] - mean) / std).fillna(0).to_numpy(np.float32))
    x_seq, x_static = sequences(train, obs), static(train)
    y = torch.tensor((train[f"obs_{var}"] - train[f"fc_{var}"]).to_numpy(np.float32))
    model = GRUModel(len(TEMP_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _epoch in range(15):
        order = torch.randperm(len(y))
        for start in range(0, len(y), 512):
            batch = order[start:start + 512]
            optimizer.zero_grad()
            loss = (model(x_seq[batch], x_static[batch]) - y[batch]).abs().mean()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return test[f"fc_{var}"].to_numpy() + model(sequences(test, obs), static(test)).numpy()


def main():
    data = pd.read_csv(CLEAN / "features.csv", dtype={"station": str}, parse_dates=["date", "issue"])
    data = data[data.model == MODEL]
    daily = pd.read_csv(CLEAN / "observations.csv", dtype={"station": str}, parse_dates=["date"])
    obs = {}
    for station, rows in daily.groupby("station"):
        rows = rows.set_index("date").asfreq("D")
        table = pd.DataFrame({"tmax": rows.tmax, "tmin": rows.tmin, "rain": np.log1p(rows.prcp.shift(1))})
        old = table[table.index <= CLIMATOLOGY_END]          # scaling uses old years only
        obs[station] = ((table - old.mean()) / old.std()).fillna(0)

    lgbm = pd.read_csv(RESULTS / "predictions_temperature.csv", dtype={"station": str}, parse_dates=["date"])
    lgbm = lgbm[lgbm.model == MODEL]
    results = []
    for month in pd.date_range(TEST_START, TEST_END, freq="MS"):
        for var in ("tmax", "tmin"):
            rows = data.dropna(subset=[f"obs_{var}", f"fc_{var}"])
            train = rows[rows.date < month - pd.Timedelta(days=GAP_DAYS)]
            test = rows[(rows.date >= month) & (rows.date < month + pd.DateOffset(months=1))]
            if test.empty:
                continue
            results.append(test[["station", "date", "lead"]].assign(
                target=var, obs=test[f"obs_{var}"], gru=train_and_predict(train, test, obs, var)))
        print(f"GRU {month:%Y-%m} done")
    gru = pd.concat(results).merge(lgbm[["station", "date", "lead", "target", "lightgbm"]],
                                   on=["station", "date", "lead", "target"])
    table = pd.DataFrame([{"target": t, "lead": lead, "days": len(g), "MAE GRU": mae(g.obs, g.gru),
                           "MAE LightGBM": mae(g.obs, g.lightgbm)} for (t, lead), g in gru.groupby(["target", "lead"])])
    (RESULTS / "gru_vs_lightgbm.md").write_text(
        f"# GRU vs LightGBM ({MODEL}, same rows)\n\n" + table.to_markdown(index=False, floatfmt=".3f") + "\n",
        encoding="utf-8")
    print("Saved results/gru_vs_lightgbm.md")


if __name__ == "__main__":
    main()
