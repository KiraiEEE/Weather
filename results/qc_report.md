# Quality control report

53830 station-days, 12 stations, 2013-12-31 to 2026-11-02

## Values removed per rule

| rule                                      |   values removed |
|:------------------------------------------|-----------------:|
| tmax out of range (-5..52)                |                0 |
| tmin out of range (-12..38)               |                0 |
| rain out of range (0..300)                |                3 |
| tmin higher than tmax                     |                0 |
| tmax spike (12 C jump up and back)        |                6 |
| tmax stuck sensor (5 equal days)          |                0 |
| tmax very unusual for the month (|z| > 6) |               27 |
| tmin spike (12 C jump up and back)        |                8 |
| tmin stuck sensor (5 equal days)          |                0 |
| tmin very unusual for the month (|z| > 6) |                4 |

Isolation Forest marked 98 unusual days (kept, for review only).

## Valid values per station after QC

| station   |   tmax |   tmin |   prcp |
|:----------|-------:|-------:|-------:|
| Tabarka   |   4217 |   4535 |   4301 |
| Bizerte   |   4163 |   4452 |   4519 |
| Tunis     |   4168 |   4497 |   4286 |
| Kelibia   |   3964 |   4479 |   4497 |
| Jendouba  |   4234 |   4316 |   4328 |
| Enfidha   |   4218 |   4374 |   4146 |
| Kairouan  |   4248 |   4541 |   4262 |
| Gafsa     |   4279 |   4157 |   4169 |
| Sfax      |   4196 |   4468 |   4200 |
| Tozeur    |   4249 |   4416 |   4093 |
| Gabes     |   2366 |   2438 |   2425 |
| Djerba    |   4151 |   4417 |   4136 |

## Check of our SYNOP decoder: NOAA vs Ogimet on the same days

| station   | variable   |   days |   same_value_% |
|:----------|:-----------|-------:|---------------:|
| Sfax      | tmax       |     78 |          100.0 |
| Sfax      | tmin       |     78 |          100.0 |
| Sfax      | prcp       |     81 |          100.0 |
| Tunis     | tmax       |     80 |          100.0 |
| Tunis     | tmin       |     83 |          100.0 |
| Tunis     | prcp       |     84 |          100.0 |
| Kairouan  | tmax       |     81 |          100.0 |
| Kairouan  | tmin       |     78 |          100.0 |
| Kairouan  | prcp       |     80 |          100.0 |
| Gabes     | tmax       |     82 |          100.0 |
| Gabes     | tmin       |     84 |          100.0 |
| Gabes     | prcp       |     84 |          100.0 |
| Djerba    | tmax       |     76 |          100.0 |
| Djerba    | tmin       |     81 |          100.0 |
| Djerba    | prcp       |     81 |          100.0 |
| Tozeur    | tmax       |     78 |          100.0 |
| Tozeur    | tmin       |     77 |          100.0 |
| Tozeur    | prcp       |     76 |          100.0 |
| Gafsa     | tmax       |     73 |          100.0 |
| Gafsa     | tmin       |     66 |          100.0 |
| Gafsa     | prcp       |     66 |          100.0 |
| Bizerte   | tmax       |     81 |          100.0 |
| Bizerte   | tmin       |     84 |          100.0 |
| Bizerte   | prcp       |     84 |          100.0 |
| Kelibia   | tmax       |     76 |          100.0 |
| Kelibia   | tmin       |     82 |          100.0 |
| Kelibia   | prcp       |     82 |          100.0 |
| Jendouba  | tmax       |     76 |          100.0 |
| Jendouba  | tmin       |     71 |          100.0 |
| Jendouba  | prcp       |     66 |          100.0 |
| Tabarka   | tmax       |     82 |          100.0 |
| Tabarka   | tmin       |     83 |          100.0 |
| Tabarka   | prcp       |     83 |          100.0 |
| Enfidha   | tmax       |     81 |          100.0 |
| Enfidha   | tmin       |     78 |          100.0 |
| Enfidha   | prcp       |     80 |          100.0 |
