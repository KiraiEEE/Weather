"""Project settings: paths, stations, dates and model parameters."""
from pathlib import Path

# folders
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"          # download cache
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
for folder in (RAW, CLEAN, RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 42

# stations
# WMO id: (name, latitude, longitude, elevation in m). Sfax is the main station.
STATIONS = {
    "60750": ("Sfax", 34.718, 10.691, 26),
    "60715": ("Tunis", 36.851, 10.227, 7),
    "60735": ("Kairouan", 35.667, 10.100, 68),
    # NOAA's files say (33.734, 9.919, 125 m) from 2024 on, but temperatures show no jump against Djerba/Sfax
    # that year, and GHCN also gives the coast: treated as a metadata change, not a real move.
    "60765": ("Gabes", 33.877, 10.103, 8),
    "60769": ("Djerba", 33.875, 10.775, 6),
    "60760": ("Tozeur", 33.940, 8.111, 87),
    "60745": ("Gafsa", 34.422, 8.823, 323),
    "60714": ("Bizerte", 37.245, 9.791, 6),
    "60720": ("Kelibia", 36.850, 11.083, 30),
    "60725": ("Jendouba", 36.483, 8.800, 144),
    "60710": ("Tabarka", 36.980, 8.877, 70),
    "60731": ("Enfidha", 36.083, 10.433, 6),
}
MAIN_STATION = "60750"

# observations
NOAA_FIRST_YEAR = 2014          # NOAA Global-Hourly history
NOAA_LAST_DAY = "2025-08-24"    # NOAA stopped receiving Tunisian data after this day
OGIMET_START = "2025-06-01"     # Ogimet raw SYNOP; overlaps NOAA for a cross-check
WET_DAY_MM = 1.0                # a "rain day" means >= 1 mm (WMO definition)
HEAVY_RAIN_MM = 10.0

# forecasts
# label: (Open-Meteo model name, first day with archived forecasts)
MODELS = {
    "IFS": ("ecmwf_ifs025", "2024-02-10"),          # ECMWF physics model
    "AIFS": ("ecmwf_aifs025_single", "2025-02-25"),  # ECMWF AI model
    "GFS": ("gfs_seamless", "2024-02-10"),           # NOAA physics model
}
FORECAST_END = "2026-10-25"
MAX_LEAD_TEMP = 7               # days ahead for temperature
MAX_LEAD_RAIN = 6               # rain needs one extra day of data, and the archive stops at day 7

# evaluation
CLIMATOLOGY_END = "2023-12-31"  # climatology uses only years before any forecast/test data
TEST_START = "2025-09-01"       # scored period (same for every model)
TEST_END = "2026-10-25"
FIRST_PREDICTION = "2025-03-01"  # earlier months are predicted only to calibrate the intervals
MIN_TRAIN_ROWS = 2000
GAP_DAYS = MAX_LEAD_TEMP + 1    # training data must end this many days before the test month
INTERVAL = 0.8                  # 80% prediction intervals (10th to 90th percentile)

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
