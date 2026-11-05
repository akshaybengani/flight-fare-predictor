# Flight price data scout: which datasets can train a ticket-price model

Scope: download and measure candidate datasets for predicting airline ticket price over time. Nothing in `src/`, `templates/`, `models/`, the README, or the existing `data/` folders was changed. Nothing was committed.

Question per dataset: does it have **repeated prices for the same flight at different observation dates**, a **search/observation date** separate from the **flight date**, and which **drivers** (weekday, holidays, weather, fuel, crisis, events, seasonality)?

All numbers below were measured on the downloaded files with DuckDB or pandas. "Sanity" checks are plausibility tests only. Nothing here proves a source is real.

---

## 1. Summary table

| # | Dataset (folder) | Rows | Date range (observation / flight) | Repeated prices? | Drivers present | Verdict |
|---|---|---|---|---|---|---|
| 1 | US Airfare Time Machine (`us-airfare-time-machine`) | 38.6M quotes (1.68M route-day aggregates) | search 2026-03-12 to 2026-09-30 (203 days); flight 2026-03-19 to 2027-03-29 (376 days) | **Yes**: 1.07M itineraries, 90% seen on 2+ search days, mean 36 search days each | weekday, holiday (derivable); no weather/fuel/events | **USEFUL** (rank 1) |
| 2 | Expedia flightprices (`expedia-flightprices`) | ~82M (CSV 31 GB; 1.64M row 2% sample saved) | search 2022-04-16 to 2022-10-05 (171 days); flight 2022-04-17 to 2022-11-19 (217 days) | **Yes**: ATL-BOS and LAX-SFO fully scanned, 49,797 flights, mean 16.9 search days each, 83% on 2+ days | weekday, holiday (derivable); no weather/fuel/events | **USEFUL** (rank 3, data is old, US only) |
| 3 | Daniil Baltic/Europe flights (`extra-historical-daniil`) | 20.5M | observed 2025-05-02 to 2026-10-08 (212 days); flight 2025-05-03 to 2026-11-16 | **Yes**: 1.97M itineraries, 91% seen on 2+ days, mean 10.4 obs days each, 97% of those change price | weekday, holiday (derivable); no weather/fuel/events | **USEFUL** (rank 2) |
| 4 | Flight price dataset, flight-prices-by-date (`flight-prices-by-date`) | 317,260 | searched 2024-06-27 to 2024-07-03 (7 days); flight 2024-07-03 to 2024-12-11 | Yes, but thin: 4.8 search days per itinerary, only 7 search days | weekday (derivable) | PARTLY USEFUL |
| 5 | DB1B US market fares (`bts-db1b`, 6 quarters 2024 Q1 to 2025 Q2) | 47.5M market rows total (7.3-8.5M per quarter) | quarterly only; no search date, no flight date | No | quarter (seasonality, 3-month steps) | PARTLY USEFUL (benchmark only) |
| 6 | Lumnikalab European fare index (`extra-fare-history-lumnikalab`) | 5.46M obs, aggregated to 181K route-months | observed 2026-06-14 to 2026-11-03; departure months | Over time but aggregated, no per-flight rows | booking window, weekday, month | PARTLY USEFUL |
| 7 | German domestic fares (`german-fares`) | 62,627 | scrape 18-24 Oct 2019 (7 days); flight Oct-Dec 2019 | No: 1.00 scrape day per flight | days_left via horizon label | NOT USEFUL |
| 8 | NZ airfares (`nz-fares`) | 162,833 | travel 1 Oct to 9 Dec 2019; no scrape date | No observation date | none | NOT USEFUL |
| 9 | Goibibo India (`goibibo-india`) | 300,261 | flight dates 1-31 Jul 2023; no scrape date | No observation date | weekday | NOT USEFUL |
| 10 | Bangladesh (`bangladesh-fares`) | 57,000 | departure 2025-01-03 to 2026-03-31; no scrape date | No | none | NOT USEFUL (looks synthetic) |
| 11 | Oil/fuel vs airfare (`airfare-vs-oil-fuel`) | 14,850 (monthly, 90 months) | 2019-01 to 2026-06, monthly | No | fuel, crisis labels, events | NOT USEFUL (looks synthetic) |
| 12 | Priceline Dhaka (`priceline`) | 2,517 | no date | No | none | NOT USEFUL |
| 13 | 57 booking sites (`extra-57-sites-ktemia`) | 18,299 | single snapshot; 6 departure dates in Oct-Dec 2026 | No (repeats are across providers, not time) | none | NOT USEFUL for time |

---

## 2. Ranking (useful ones)

1. **US Airfare Time Machine** (`us-airfare-time-machine`). Best combination of repeats, scale, horizon and realism. Repeated quotes across 203 search days and 376 departure dates. A clear booking-window curve and a Sunday premium. Route distance explains route price (r = 0.88 across 50 routes). Download is 222 MB, so it is easy on a MacBook.
2. **Daniil Baltic/Europe** (`extra-historical-daniil`). Second-best. Apache-2.0 licence, 18 months of observation, 20.5M rows. Strong price-to-duration correlation (r = 0.66). Caveat: the median within-itinerary price range is 91%, which is unusually volatile, so check whether one `itinerary_id` mixes fare products before trusting it.
3. **Expedia flightprices** (`expedia-flightprices`). Largest and realistic (fare vs distance r = 0.49 in the sample), with 82M rows and an observation date per row. Limits: data is from 2022 and the full repeat check covered only two routes. Hard to use on a laptop: 31 GB CSV, so use the 2% sample or stream it.

Next tier: flight-prices-by-date (PARTLY), DB1B (PARTLY, benchmark), Lumnikalab (PARTLY, aggregate index).

---

## 3. Per-dataset detail

### 3.1 US Airfare Time Machine (`us-airfare-time-machine`): USEFUL, rank 1

Files: `flight_prices.parquet` (38,584,829 quotes, 99 MB), `daily_fares.csv` (1,683,033 rows, 133 MB), `routes.csv` (50 routes), `airports.csv` (16 airports).

- **Repeats:** 1,073,215 itinerary IDs. 90.4% are seen on two or more search dates. Mean 36 search days per itinerary (median 25, max 173). Mean distinct prices per itinerary is 8.0. Flight-key repeats (airline + route + date + departure time): 92.9% on two or more search days.
- **Observation vs flight date:** both present. `days_before_departure` ranges 7 to 180 and matches `departure_date - search_date` in every row (0 mismatches), so days_left is not confounded.
- **Coverage gaps:** some search days are partial (2 Apr, 3 Apr, 6 May, 11-26 Jun have roughly half the usual volume). Model these as missing, not as zero.
- **Drivers:**
  - Weekday: present. Median price by departure weekday ranges from about 163 (Tue) to 251 (Sun).
  - Holidays: derivable from departure date (US federal and school holidays need a join; not downloaded here).
  - Weather, oil/fuel, conflict, events: absent.
  - Seasonality: 13 departure months (Mar 2026 to Mar 2027).
- **Booking-window curve (median fare, USD):** 293 at 7 days out, 246 at 14 days, 205 at 21 days, 193 at 30 days, about 184 at 45-60 days, about 190-197 at 90-180 days. Classic U-shape, which is a good sign the data is real.
- **Sanity:**
  - Route distance vs route median price: r = 0.89 across 50 routes.
  - Duration vs price: r = 0.25 across flights.
  - Stops vs price: r = -0.02 (weak, as expected for a mix of routes).
  - Price range $19 to $5,046, median $176.
- **Licence:** Kaggle lists "other". Check the dataset page before any commercial use.
- **Use on MacBook:** the parquet loads fine. Use DuckDB or pandas with column selection.

### 3.2 Expedia flightprices (`expedia-flightprices`): USEFUL, rank 3

Files: `itineraries.csv` (31.1 GB, not loaded whole), `sample.parquet` (1,641,725 rows, 173 MB, a 2% Bernoulli sample I saved).

- **Scale:** the full row count was not counted in full. The sample and the Kaggle description suggest about 82M. Flag as not verified.
- **Coverage:**
  - Search dates: 2022-04-16 to 2022-10-05 (171 days in the sample).
  - Flight dates: 2022-04-17 to 2022-11-19 (217 days).
  - 235 origin-destination pairs. Mean fare 340 USD, range 19.6 to 4,763.
- **Repeats:** full-file scan of two routes only (ATL-BOS and LAX-SFO). 49,797 flights. Mean 16.9 search days per flight, median 9, 83% on two or more days, mean 4.5 distinct prices per flight. Estimated across all routes, not verified.
- **Drivers:**
  - Weekday: present. Sunday mean fare 400 USD, Tuesday 297.
  - Non-stop vs connecting: 251 vs 373 USD mean.
  - Holidays: derivable. No weather, fuel, or events.
- **Sanity:** fare vs distance r = 0.49, fare vs duration r = 0.43, distance vs duration r = 0.71 (all on the 2% sample).
- **Licence:** Attribution.
- **Use on MacBook:** the 2% sample is easy. For the full file, stream with DuckDB; it took about 80 s for the sample pass.

### 3.3 Daniil Baltic/Europe (`extra-historical-daniil`): USEFUL, rank 2

File: `flights_clean.parquet` (20,496,381 rows, 625 MB). Found via search; not in the original list.

- **Repeats:** 1,967,106 itineraries. 91.4% seen on two or more observation days. Mean 10.4 observation days per itinerary (median 8, max 32). Mean 10.4 rows each.
- **Price movement:** of itineraries with two or more observations, 96.97% change price. Median relative range (max-min)/min is 91%. Verify before relying on it: check whether `itinerary_id` combines different fare products.
- **Observation vs flight date:** `price_datetime` (observation) and `departure_datetime` (flight). `days_to_departure` 1 to 176.
- **Coverage:** observations 2025-05-02 to 2026-10-08 (212 days). Departures 2025-05-03 to 2026-11-16. 1,096 routes, 59 airlines.
- **Drivers:**
  - Weekday: present. Saturday and Sunday mean around 260 EUR, Tuesday and Wednesday around 230.
  - Holidays: derivable. Europe holiday calendars are free.
  - Weather, fuel, events: absent.
  - Seasonality: 18 months.
- **Sanity:** price vs travel time r = 0.66, vs emissions r = 0.59, vs days-to-departure r = -0.15 (expected sign).
- **Licence:** Apache-2.0.
- **Use on MacBook:** 625 MB parquet, read with column selection.

### 3.4 flight-prices-by-date (`flight-prices-by-date`): PARTLY USEFUL

File: `flights.csv` (317,260 rows, 30 MB). US domestic, 30 routes, 46 airlines.

- Searched 2024-06-27 to 2024-07-03 (7 days). Departures 2024-07-03 to 2024-12-11 (12 departure dates in the file).
- Repeats: 85.7% of itinerary keys (airline + route + departure date) seen on two or more search days. Mean 4.8 search days per key.
- Days before departure 6 to 167 (median 15). Weekday counts are skewed (147K rows on one weekday), because a few departure dates dominate.
- Median price $316.
- Limits: 7 search days is too short to learn price movement across a season.

### 3.5 DB1B US market fares (`bts-db1b`): PARTLY USEFUL, benchmark only

- Download method: the BTS PREZIP URLs work with plain `curl` (no web form). Downloaded and verified: `DB1BMarket_2024_1` to `2025_2` (six quarters, zips kept in `data/bts-db1b/`). The 2025 Q2 transfer stalled once and was resumed with `curl -C -`; its final size matches the server's Content-Length.
- Per-quarter results (all measured on the full market file):

| Quarter | Rows | OD pairs | Median MktFare (USD) | Corr(fare, distance) | Passengers |
|---|---|---|---|---|---|
| 2024 Q1 | 7,409,931 | 63,563 | 228.0 | 0.290 | 13.9M |
| 2024 Q2 | 8,533,978 | 67,848 | 234.5 | 0.318 | 16.3M |
| 2024 Q3 | 8,297,869 | 69,873 | 221.5 | 0.302 | 15.9M |
| 2024 Q4 | 8,525,077 | 68,114 | 237.0 | 0.295 | 15.9M |
| 2025 Q1 | 7,297,028 | 65,848 | 234.0 | 0.283 | 13.8M |
| 2025 Q2 | 8,450,420 | 70,249 | 230.6 | 0.296 | 16.1M |

- Quarter-to-quarter medians vary only 221 to 237 USD, with stable fare-distance correlation (0.28 to 0.32). Seasonality is visible only at quarter granularity.
- `DB1BMarket_2024_1`: 7,409,931 market rows (63,563 origin-destination pairs). Median `MktFare` 228 USD. Fare vs distance r = 0.29. Passengers 13.9M.
- No search date and no flight date. Repeats are not possible.
- Use as an external fare-level benchmark, not as the training set.
- `DB1BTicket` (itinerary-level) was not downloaded.

### 3.6 Lumnikalab European fare index (`extra-fare-history-lumnikalab`): PARTLY USEFUL

Found via search. Only aggregates are published.

- `summary.json`: 5,455,152 observations, 96,360 routes, observed 2026-06-14 to 2026-11-03.
- `fares-by-month.csv`: 181,250 route-month rows (median, p10, min per route and departure month).
- Research tables: monthly fare index, weekday effect, booking-window premium. Booking window: average premium 49% at 0-3 days out, falling to 17% at 1-1.5 months out.
- Licence: Attribution. Source and collection method not stated.
- Useful as a time-series feature or sanity benchmark. Not usable for per-flight modelling.

### 3.7 German domestic fares (`german-fares`): NOT USEFUL

- 62,627 rows. Scrape date 18-24 Oct 2019 (7 days). Departures Oct to Dec 2019.
- `departure_date_distance` gives horizon buckets (1 week to 6 months). Each flight key appears on exactly one scrape date, so there are no repeats.
- Price min 26, median 164 EUR. One row has a corrupted price (13.5M), so clean the price column before use.
- Licence: Attribution.

### 3.8 NZ airfares (`nz-fares`): NOT USEFUL

- 162,833 rows, 32 routes. Travel dates 1 Oct to 9 Dec 2019. No scrape or search date.
- 1,841 exact duplicate rows.
- Within a travel-date key, the same price appears multiple times. These are not observations over time.
- Licence: Attribution.

### 3.9 Goibibo India (`goibibo-india`): NOT USEFUL for time dynamics

- 300,261 usable rows (file has 314,914 lines). 30 Indian city routes. Flight dates 1-31 Jul 2023.
- No search or observation date. Flight keys repeat across flight dates, which is cross-section, not price-over-time.
- Prices are strings with commas (e.g. "6,013"); parsed.
- Licence: CC-BY-NC-SA-4.0 (non-commercial only).

### 3.10 Bangladesh (`bangladesh-fares`): NOT USEFUL, looks synthetic

- 57,000 rows, 152 routes, 24 airlines. Departures 2025-01-03 to 2026-03-31. No observation date.
- Class shares are almost exactly equal: Economy 33.5%, First Class 33.3%, Business 33.2%.
- Total fare does not equal base + tax (max residual 93,165 BDT).
- Direct flights have median total 12,229 BDT vs 78,614 (1 stop) and 82,076 (2 stops). Implausible.
- Duration vs total fare r = 0.35.
- Licence: CC-BY-SA-4.0.

### 3.11 Oil/fuel vs airfare (`airfare-vs-oil-fuel`): NOT USEFUL, looks synthetic

Six CSVs. Monthly, 2019-01 to 2026-06 (90 months).

- `airline_ticket_prices.csv`: `total_fare_usd` equals `base + fuel_surcharge + taxes` exactly (max residual 0.000). `avg_route_km` is one fixed value per airline and route class.
- Brent vs jet fuel correlation 0.996. Monthly Brent change has standard deviation 0.21, which looks generated.
- `conflict_oil_events.csv` (40 rows) has hand-typed impact percentages and no source.
- Conflict phase labels (e.g. "US-Iran War", "Stabilisation") are assigned, not measured.
- The 2026 oil values (up to 169 USD) could not be checked against EIA here.
- Licence: CC-BY-SA-4.0.

### 3.12 Priceline Dhaka (`priceline`): NOT USEFUL

- 2,517 rows. Departures from Dhaka (DAC). No date of any kind.
- Licence: CC0-1.0.

### 3.13 57 booking sites (`extra-57-sites-ktemia`): NOT USEFUL for time

- 18,299 rows. 10 routes, 57 providers, 6 departure dates (13 Oct to 5 Dec 2026).
- Single snapshot, no observation time. Repeats are across providers, not over time.
- Useful for price dispersion across sites only. Median 429 USD. Duration vs price r = 0.14.
- Licence: MIT.

---

## 4. Checks not done (say so)

- **Search results not evaluated in detail:** `yiclgg/airfare-database` (4.4 MB, 2025) and `ktemia` were noted but not fully checked. `daniilchuhai/flights-test` is the same dataset as the Daniil set above.
- **Expedia full-file counts:** total rows, and repeat stats for all routes, were not computed. Only the sample and two routes.
- **Holiday, weather, fuel joins:** not downloaded. Options below.
- **Realism:** no ground truth available. Verdicts rest on plausibility tests (distance, booking-window shape, weekday effects, accounting identities).
- **Licences:** only Kaggle-listed labels were checked; the US dataset is "other".

---

## 5. Recommendation

**Build the next model on:**

1. **US Airfare Time Machine** (`us-airfare-time-machine`). Primary. Largest repeat coverage over 13 departure months with an observation date for every quote. Use `flight_prices.parquet`.
2. **Daniil Baltic/Europe** (`extra-historical-daniil`). Second dataset to test generalisation across markets. Confirm the `itinerary_id` fare-product question first.

Use Expedia (2022) as a fallback or for a third market check.

**External data to join (free; none downloaded in this session):**

| Driver | Source to join | Joins on | Notes |
|---|---|---|---|
| Holidays (US) | `holidays` Python package, or US federal + school calendars | departure date | Add days-to-holiday and holiday-weekend flags |
| Holidays (Europe) | `holidays` package per country | departure date and origin/destination country | For the Daniil set |
| Weather | Meteostat (free) or NOAA GHCN daily | airport lat/lon (in `airports.csv`) x date | Use only if weather is a hypothesis; weak for price |
| Jet fuel / oil | EIA (US Gulf Coast jet fuel, weekly/monthly; Brent daily) | observation date or month | Price moves after oil shocks |
| Events / crisis | GDELT or a hand-curated list with dates | observation date | Keep the list dated and sourced |

---

## 6. Downloads and disk

- Downloaded successfully: all 9 brief candidates (US Time Machine, oil/fuel, German, NZ, Bangladesh, Goibibo, flight-prices-by-date, Priceline, Expedia), the three extra Kaggle datasets (Daniil, Lumnikalab, ktemia), and BTS DB1B market quarters 2024 Q1 and Q2.
- BTS DB1B quarters 2024 Q1 to 2025 Q2 are all downloaded (six zips). `DB1BTicket` (itinerary-level) was not attempted. To fetch more quarters: `curl -fL -C - -o data/bts-db1b/DB1BMarket_<YEAR>_<Q>.zip https://www.transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BMarket_<YEAR>_<Q>.zip`.
- The extracted CSVs (9.4 GB) were deleted after measuring; the zips are the source.
- Expedia used the Kaggle CLI; the downloaded zip was unzipped to a 31 GB `itineraries.csv`.
- A first attempt to download the small datasets silently failed because the shell loop did not split arguments under zsh. It was re-run under bash and all seven succeeded.
- Disk used by the new folders (approximate): US 222 MB, Expedia ~29 GB, Daniil 625 MB, DB1B 605 MB (six zips; extracted CSVs deleted), Goibibo 26 MB, flight-prices-by-date 29 MB, German 7 MB, NZ 13 MB, Bangladesh 13 MB, oil 4 MB, Priceline 1 MB, ktemia 3 MB, Lumnikalab 13 MB. Total roughly 32 GB. Free disk after downloads: about 30 GB.
- Existing folders in `data/` (`hf-atlanta`, `airfare-ml`, etc.) were not touched.
