# Enrichment report: do free external features improve the fare model?

**Short answer: barely, and not where it matters most.** 121 new real-data columns (calendar, holidays, festivals, cricket, weather, fuel/FX, airport statics) moved the same-period score by +0.17 points, *lowered* the recalibrated score by 0.41, and raised the unseen-period score by +1.03. Only the last is outside noise, and it comes almost entirely from the calendar group.

All numbers are measured in this repo (`src/train_v2.py`, `models/experiments_v2.json`, `models/metrics_v2.json`). Accuracy = 100 - MAPE, same definition and same three scenarios as `src/train.py`.

## Headline (final config, full `train.py` GBM)

| scenario | baseline (v1, `metrics.json`) | best v2 (`+all_booking_time`) | change |
|---|---|---|---|
| A same-period random 20% | 93.04 | 93.21 | +0.17 |
| E recalibrated (2022 + 10% of 2023 flight-days) | 87.00 | 86.59 | -0.41 |
| C unseen period (train 2022, test 2023) | 66.17 | 67.20 | +1.03 |

Within-10% / MAE (INR): A 78.8% / 1425 (v1 77.6% / 1500); E 53.4% / 2814 (v1 55.1% / 2717); C 20.3% / 5806 (v1 19.9% / 5889).

`+all_booking_time` = calendar + weather climatology + fuel/FX + airport statics, i.e. every feature that is knowable when the customer books. It was picked because it is the best honest rung on A; it is not the best on E or C individually (see below), so "best configuration" depends on which scenario you care about.

## Ablation ladder (fast GBM: lr 0.1, 1500 iterations, otherwise identical to train.py)

The ladder uses a faster learning rate so nine rungs fit in the time available; its baseline lands within 0.1-0.2 points of the full-config baseline (92.95 / 86.82 / 66.07 vs 93.04 / 87.00 / 66.17). Cells: accuracy % / within-10% / MAE INR. `n` = new columns that vary in that scenario's training data.

| rung | A same-period | E recalibrated | C unseen period |
|---|---|---|---|
| baseline | 92.95 / 77.3 / 1514 | 86.82 / 54.5 / 2749 | 66.07 / 20.1 / 5893 |
| baseline, different GBM seed (noise floor) | 92.91 / 77.2 / 1526 | 86.82 / 54.6 / 2769 | 65.89 / 20.0 / 5926 |
| + calendar, holidays, festivals, cricket (n 36-39) | 92.92 / 77.3 / 1509 | 86.53 / 53.2 / 2804 | **67.31** / 20.6 / 5758 |
| + weather, climatological (n 14-16) | 92.97 / 77.5 / 1492 | 86.44 / 52.7 / 2803 | 66.70 / 20.1 / 5856 |
| + weather, ACTUAL (upper bound) (n 18-20) | 92.92 / 77.4 / 1490 | 86.22 / 52.1 / 2836 | 66.33 / 19.9 / 5878 |
| + fuel / FX (n 9-18) | 92.93 / 77.3 / 1501 | 86.81 / 54.1 / 2781 | 65.75 / 20.2 / 5905 |
| + airport / city static (n 14) | 92.87 / 76.8 / 1540 | 86.86 / 54.8 / 2732 | 65.97 / 19.9 / 5916 |
| + all booking-time (no actual weather) (n 73-87) | **93.08** / 78.1 / 1446 | 86.35 / 52.6 / 2859 | 67.29 / 20.6 / 5760 |
| + all, incl. actual weather (n 93-107) | 93.02 / 77.9 / 1465 | 86.13 / 51.8 / 2891 | **67.43** / 20.5 / 5766 |

Noise floor: changing only the GBM seed moved accuracy by 0.04 (A), 0.00 (E) and 0.18 (C) points. I did not run multiple seeds per rung, so treat differences below about 0.2 points as indistinguishable from zero.

## Reading the table

- **A (same period): nothing.** Every single family is within +-0.1 of baseline; the combination gains 0.1-0.2 at best. Within one scrape the model already sees the other days of the same flights, and `days_left` plus `dow` already identify each of the 50 journey dates. A date-level feature (holiday, rain, fuel price) cannot say anything a date identifier does not.
- **E (recalibrated): slightly worse.** Calendar, weather and the combos lose 0.3-0.7 points; fuel and static are neutral. With ~100 extra date-level columns the trees can memorise 2022's date fingerprints and mis-transfer them to 2023's different dates.
- **C (unseen period): calendar helps, +1.2 points on the ladder (+1.0 at full config).** It is the only family clearly above noise. Weather climatology gives about +0.6 (borderline), fuel/FX and statics nothing. Importance in scenario C points at `fest_days_since_major`, `fest_days_to_next_major` and the climatological precipitation, all small (0.1 point each when shuffled).
- **Why gains are small, the structural reason.** Each scrape has one observation date and about 50 journey dates, so each external feature takes at most ~50 distinct values per period. Festivals barely appear in the window: the sample covers 12 Feb - 1 Apr 2022 and 16 Jan - 6 Mar 2023. Diwali, Dussehra, Durga Puja, Onam, Ganesh Chaturthi, Eid, Christmas and school vacations never occur (12 festival/season columns are constant and automatically dropped; see `docs/ENRICHED_FEATURES.md`). Only Holi 2022, Maha Shivratri, Valentine's Day and the Republic Day tail are inside it, and the only IPL window (from 26 Mar 2022) covers six journey days. Holiday effects cannot be learned from data that contains almost no holidays. This is a data-coverage limit, not a feature-engineering one.
- **Fuel/FX is confounded with period.** The observation-date columns (`fx_*_obs`) are constants within a scrape, i.e. a duplicate of `period`. Journey-date fuel varies within 50 days but those prices are the realised price on the flight date, not the booking date. The 2022-to-2023 fare gap (economy about 55% higher) is not explained by a two-point fuel series; one data point per period cannot identify a price-vs-fuel relationship.
- **Airport statics are 7 values each**, a re-labelling of `source`/`destination`. Result: zero, as expected.

## Which new features mattered (held-out permutation importance, `models/importance_v2.json`)

Measured on fast-config models (the full-config model is too slow to shuffle ~90 columns), 8000 held-out rows, 2 shuffles; scenario E importance was not computed.

- **Scenario A:** shuffling whole groups drops accuracy by weather climatology 3.96 pts, fuel/FX 2.13, calendar 1.59, airport static 0.09. Top single columns: `wxl_o_tmin` 0.83, `wxl_o_tmax` 0.79, `wxl_*_gust_max` ~0.59, `fx_jet_usd_gal` 0.43, `cal_o_days_to_next_holiday` 0.34.
  **Do not read this as signal.** The ablation shows removing these groups changes accuracy by ~0. They are used as stand-in date identifiers (a temperature climatology is a smooth function of the calendar day), duplicating `days_left`/`dow`. Permutation importance measures reliance, not added information; the ablation is the evidence that matters.
- **Scenario C:** all groups ~0 except calendar 0.17: `fest_days_since_major` 0.14, `wxl_o_precip_mm` 0.10, `fest_days_to_next_major` 0.07.

## Actual vs lagged weather (leakage caveat)

- `wxa_*` is the weather that actually occurred on the journey date. It is **not knowable at booking time** (bookings here are 1-50 days ahead), so a deployed model could only use a forecast. Results with it are an upper bound.
- `wxl_*` is the 2015-2021 mean for the same calendar day +-3 days at each airport: legitimate at booking time.
- Result: actual weather did *not* beat climatology on any scenario (A 92.92 vs 92.97; E 86.22 vs 86.44; C 66.33 vs 66.70). Even with hindsight, weather on the day does not explain fare variation here. Reasons: fares are set at booking time, 1-50 days earlier; this is a Feb-Mar dry season (mean daily precipitation under 0.3 mm at 6 of 7 airports; Delhi's fog is the main event and ERA5 has no visibility, so fog is only a humidity/wind proxy); and the scrape is one booking day per flight, so weather-driven repricing is invisible.
- Also not done: weather "as of the observation date" (it is constant per scrape, so equivalent to a period flag) and any real forecast archive (none is free for this period).

## Recommendation

1. **Do not claim the model got "smarter".** On the realistic scenarios (A, E) the enrichment is a wash (+0.17 / -0.41), and v2 is 100+ columns heavier for it. Keep `fare_model.joblib` as the serving model. `fare_model_v2.joblib` is provided for comparison and is not wired into the app.
2. If one family is kept, keep **calendar + festival distance features** (the only family above noise, on the unseen-period scenario). Drop actual weather (leaky, no gain), airport statics (redundant), and the fuel-observation columns (duplicate `period`).
3. The real bottleneck is **data coverage in time, not feature richness**: one observation date per scrape and ~100 journey dates in total. To learn festival, holiday, fuel or weather effects you need repeated price observations per flight across many booking dates and across Diwali/Dussehra/summer/wedding seasons. Those features are ready (calendar tables cover 2021-10 to 2024-03; Open-Meteo and FRED cache can extend), so they will be useful once such data exists.
4. Weather is non-commercial only under Open-Meteo's free tier (see `docs/ENRICHED_FEATURES.md`); do not ship it commercially without a paid plan. Given it gave no gain, there is little reason to.

## Caveats and things not done

- **DGCA monthly traffic by city pair/airline: skipped.** Not fetched; no reliable machine-readable public endpoint found in the time available.
- Holiday dates (especially lunar/Islamic) come from the `holidays` package's computed dates and may be a day off the gazette. School-vacation and wedding-season flags are **hand-set approximations**, not a dataset (and are constant in this sample except exam season and wedding season, both collinear with period).
- Cricket features come from Cricsheet's played-match records (not pre-published schedules) restricted to men's IPL/internationals; sparse in this window. The Women's Premier League and non-metro hosts are not represented.
- Jet fuel is the US Gulf Coast kerosene-type price (EIA via FRED) in USD, a proxy for Indian ATF, which is set monthly by oil companies; the real Indian ATF series was not available keyless. USD/INR and Brent also from FRED. A custom User-Agent is rejected by FRED, so the script uses the default requests UA for that host.
- City population values are Wikidata's Census 2011 entries (city proper, not metro area). OpenFlights route counts are a c.2014 snapshot.
- The scrape date for 2022 (11 Feb) is assumed in `build_dataset.py`; if wrong, the journey-date features shift with it.
- Single seed per rung, single split per scenario; differences below ~0.2 points are not distinguishable from noise. The ladder used a faster GBM than the final numbers. Permutation importance came from fast-config models on scenarios A and C only (E skipped). The final-config E run and the importance step were re-run after an earlier, slower variant was stopped; the code in `train_v2.py` is the version that produced the saved files.
- Reproducibility: `python src/enrich.py` was run end-to-end against an empty `data/external/` (in a scratch copy) and produced a byte-identical set of columns/values to the in-repo `data/enriched.parquet`.

## Files

`src/enrich.py`, `src/train_v2.py`, `data/enriched.parquet` (gitignored), `data/external/*` (gitignored caches), `docs/ENRICHED_FEATURES.md`, `models/fare_model_v2.joblib`, `models/metrics_v2.json`, `models/experiments_v2.json`, `models/importance_v2.json`. v1 model files and `src/app.py` were not touched.
