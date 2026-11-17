# FareSight: design notes

Working name **FareSight**, tagline "Fare intelligence for flight booking". The name is a placeholder: no trademark or registered status is claimed.

The brief was "make it look hand-made, not generated". That shaped most decisions below: fewer boxes, one accent colour, a real typographic voice, and illustration drawn for this product rather than dropped in.

## Palette (light only)

| Token | Hex | Used for |
|---|---|---|
| `--paper` | `#F5F0E6` | page background (warm off-white) |
| `--paper-deep` | `#ECE5D6` | wells, table stripes, hero band |
| `--sheet` | `#FBF8F1` | ticket and panel fill (still not pure white) |
| `--ink` | `#13233D` | text, primary buttons, chart lines (deep navy) |
| `--ink-2` | `#4B5870` | secondary text |
| `--ink-3` | `#636D82` | helper text, axis labels |
| `--rule` / `--rule-strong` | `#D8CFBC` / `#BFB49D` | hairlines, dashed dividers, illustration strokes |
| `--accent` | `#D2452B` | the one accent, vermilion: plane glyphs, active tab, the predicted point, focus underline |
| `--down` | `#2B6B4E` | meaning only: predicted fare lower than listed |
| `--up` | `#B93A22` | meaning only: predicted fare higher than listed |

Colour is used only for meaning. Green and vermilion appear on the price difference and nowhere decorative. The accent doubles as the "up" tone, which is why it stays rare.

## Type

- **Fraunces** (variable, upright and italic) for headings, flight codes, times and prices. A soft old-style serif with enough character to read as chosen.
- **Source Sans 3** (variable) for body, labels and tables. A clean humanist sans.
- Tabular numerals (`font-variant-numeric: tabular-nums`) on every price, time and table figure.
- Both are SIL Open Font License 1.1. Files are self-hosted in `static/fonts/` (woff2, latin subset, from Fontsource), so the demo needs no network. System fallbacks: `Georgia` for display, `system-ui` for body. Licence text: `static/fonts/LICENSE-OFL.txt`.

## Rhythm

8 px base, 4 px for fine gaps. Content width 1080 px. Paragraph measure capped near 62 characters.

## Layout, in words

**Search page**
- A slim header: plane mark, "FareSight" wordmark in Fraunces, two links (Search, Model & Insights).
- A hero band on warm paper with a large line-art airliner drifting along a contrail, a few outlined clouds and topographic contour lines in the corner. Headline in serif, one sentence of explanation.
- The search form is shaped like a boarding pass: From and To fields at the ends, a dashed great-circle arc with a small plane between them, then date, class and a navy "Search" button along the bottom edge.
- Under the form, before any search, a dashed placeholder where the tickets will appear.
- Results: a heading with the route and date, a single line of plain parameters, then three underlined selects (airline, stops, sort) and "Approve all on this page".
- Each flight is a ticket strip. Left: airline and flight code, large times, a route line with stop dots and a small plane. A perforation with half-circle notches. Right stub: listed fare, FareSight fare in large serif, the difference with an arrow, Approve and Details.
- Details open as a sheet under the strip: an editorial line chart on the left and a specification list of the model inputs on the right.
- Pager: plain numerals, current page underlined in the accent.
- Footer: one discreet line that predictions are model estimates from historical scraped fares, not booking advice.

**Model & Insights page**
- Same header, a serif title, tabs as underlined text (not pills): Settings, Metrics, How we got the score, Model & parameters, Data & sources, Algorithm comparison.
- Sections separated by a heavy top rule and a serif heading, not boxed cards. Metrics are large serif figures between hairlines. Charts are thin horizontal SVG bars. Tables are editorial: hairline rows, small-caps headers.
- Algorithm comparison: feature-set switch (when both result files exist), scenario switch A / E / C, ranked table with accuracy bars (current model marked with a vertical tick and highlighted row, winner flagged), a speed versus accuracy scatter with direct labels, and a size and time table. A clear "pending" state when the file is missing.

## Illustration

All drawn by hand as inline SVG symbols in `templates/base.html`: top-down airliner outline, cloud outline, contour blob, arrow glyphs. Strokes are low-contrast (`--rule`), the page stays fast. Motion: slow drift of the hero plane and a faint scroll parallax on the background. Both are disabled under `prefers-reduced-motion`.

## Loading, empty and error states

- Loading: a small plane travelling along a dashed line, with "Checking fares" beside it. No spinner.
- Nothing searched yet: a dashed placeholder ticket.
- No flights for a route, or a validation error: a grounded plane drawing and the server's own message.
- Filters leave nothing: a short line with a "clear filters" action.

## Accessibility

Visible focus ring on every control (2 px navy with a paper gap), text contrast at least 4.5:1 on paper, tabs use `role="tablist"`, the pager uses `aria-current`, and the arrows carry text so meaning never depends on colour alone.

## Files

- `static/css/faresight.css`: all styling, shared by both pages.
- `static/brand/`: logo files (see Brand below). `static/favicon.svg` is a copy of `brand/favicon.svg`.
- `static/fonts/`: Fraunces, Source Sans 3, OFL text.
- `templates/base.html`, `index.html`, `config.html`: markup and page scripts (vanilla JS, no build step).
- `src/app.py`: static folder wired up, `GET /api/algorithms` added (read-only). Everything else is unchanged.

## Algorithm comparison data

`GET /api/algorithms` reads `models/algorithms.json` (baseline 18 features) and, if present, `models/algorithms_enriched.json`. If neither exists the page shows a pending state. For development only, `FARESIGHT_MOCK=1` with `FARESIGHT_MOCK_FILE=/path/to.json` serves a mock file, marked `"mock": true` and labelled as sample data in the UI. The app never writes either results file.

## Known data quirk

Some flight codes arrive from the merged dataset as scientific notation (`6.00E-152` for `6E-152`). The UI repairs the display (`6E 152`); the API output is unchanged.

## Brand: FareSight mark and wordmark

Drawn by hand in SVG, no stock art. The mark is a folded paper plane (two navy facets with a paper-coloured fold line) flying up and to the right, trailing a short vermilion contrail that reads as the start of a rising fare curve. Flat fills, no gradients, readable at 16 px.

| File | Use |
|---|---|
| `static/brand/faresight-logo.svg` | mark and wordmark on paper background |
| `static/brand/faresight-mark.svg` | icon only, transparent background |
| `static/brand/faresight-logo-mono.svg` | single colour (navy), transparent background, for print or one-colour use |
| `static/brand/favicon.svg` | mark on a rounded paper square |
| `static/brand/faresight-mark-512.png` | 512 px export of the favicon version (made with macOS Quick Look) |

The wordmark is Fraunces at weight 600, converted to outlines, so the files render without the font installed. The site header sets the same name live in the self-hosted font.

**Colours.** Navy `#13233D` for the plane and wordmark, vermilion `#D2452B` for the contrail only, paper `#F5F0E6` as the background. In the mono version everything is navy.

**Clear space.** Keep at least the height of the plane's contrail stroke times four (about a quarter of the mark's height) free on every side.

**Minimum size.** Mark alone: 16 px. Mark with wordmark: 96 px wide (below that, use the mark alone).

**Do**
- Use the supplied files; place them on paper, white or navy-free light backgrounds.
- Use the mono file when only one colour is available.

**Don't**
- Recolour the contrail, add gradients, shadows or outlines.
- Stretch, rotate or re-set the wordmark in another typeface.
- Put the colour logo on a busy photo or a mid-tone background.
- Present the name as a trademark: FareSight is a working name with no registered status.

## Dark theme: "night cabin"

Light stays the default look. The switch (hand-drawn moon and sun, top right of the header) follows `prefers-color-scheme` on the first visit and then remembers the choice in `localStorage` (key `fs-theme`; every read and write is in try/catch, and the page works without storage). A tiny inline script in `<head>` sets `data-theme` on `<html>` before first paint, so there is no flash of the wrong theme. All colours are CSS custom properties on `:root` (light) and `:root[data-theme="dark"]`.

The dark palette is a warm, slightly green-tinted ink with parchment text and a brass accent. It deliberately avoids slate or zinc blue-black, indigo or violet, neon accents, pure black, pure white and glow.

| Token | Hex | Role |
|---|---|---|
| `--paper` | `#161816` | page |
| `--paper-deep` | `#1D201E` | wells, bar tracks |
| `--sheet` | `#202321` | tickets, boarding pass, detail sheets |
| `--ink` | `#E9E1CF` | body text (parchment) |
| `--ink-2` | `#C2BAA7` | secondary text |
| `--ink-3` | `#A8A08E` | helper text, axis labels |
| `--rule` / `--rule-strong` | `#3B3830` / `#5A5445` | hairlines, dashed dividers, input underlines |
| `--accent` | `#D9A441` | brass: contrail, active tab, predicted point, eyebrow text |
| `--up` | `#E67B61` | predicted fare higher than listed (softened vermilion) |
| `--down` | `#86BD94` | predicted fare lower than listed (sage) |
| `--btn-bg` / `--btn-ink` | `#E9E1CF` / `#161816` | primary button |
| `--art` / `--art-2` | `#363A32` / `#5A5645` | clouds, contrails, contour lines, aircraft outline |

### Contrast (WCAG 2.x, computed from the hex values)

| Pair | Dark: on page / on sheet | Light: on page / on sheet |
|---|---|---|
| Body text `--ink` | 13.7 / 12.2 | 13.8 / 14.8 |
| Secondary `--ink-2` | 9.2 / 8.2 | 6.3 / 6.8 |
| Helper `--ink-3` | 6.9 / 6.1 | 5.2 / 5.6 (after darkening to `#5A647A`) |
| Accent as text `--accent-text` | 7.9 / 7.1 | 5.0 / 5.4 (`#B93A22`) |
| Price rise `--up` | 6.3 / 5.6 | 5.0 / 5.4 |
| Price drop `--down` | 8.3 / 7.3 | 5.6 / 6.0 |
| Primary button text on button | 13.7 | 13.8 |

Every text pair passes AA (4.5:1). In light mode the bright vermilion `#D2452B` (4.0:1) is used only for non-text marks such as the plane glyph and underlines; accent text uses the darker `--accent-text`. Meaning never rests on colour alone: the arrows, the words "up" and "down", and the "Large rise / Large drop" labels carry it too.

### What was re-tinted
Boarding-pass strips and perforation notches (the notches are masks, so they show whichever ground is behind), the route line and plane glyph, contrails, clouds and contour lines, the detail chart, the algorithm bars, tick and scatter, tables, tabs, buttons, the select arrow, focus ring, and the loading plane. The mark in the header and loader is an inline symbol coloured by the same variables.

### Dark brand files
- `static/brand/faresight-logo-dark.svg`: cream plane and wordmark on `#161816`, brass contrail.
- `static/brand/favicon-dark.svg` and `favicon-light.svg`: fixed variants.
- `static/brand/favicon.svg`: one file with an embedded `prefers-color-scheme` rule, so the tab icon suits either browser theme. This is the one the pages link.

## Layout hardening (stat strips, long names, tables)

- Stat cells have `min-width: 0`, wrap long values with `overflow-wrap: anywhere` plus soft break points inside CamelCase names, and use a fluid `clamp()` size for text values (distinct from the big-number style). The grid reflows with `minmax(min(100%, 230px), 1fr)`, so it drops to fewer columns before values meet.
- Long algorithm notes and tuning descriptions fold into a "Notes" / "How it was tuned" disclosure.
- At 760 px and below, every table with a header row stacks into labelled rows instead of scrolling sideways; tickets switch to the stacked layout at the same width.
- The speed-against-accuracy scatter sizes to the viewport. On a phone it numbers the dots and lists names below, since long names cannot be labelled in place.
- Audited at 1440, 1024, 768 and 390 px in the dark theme on every Model & Insights tab with a script that checks for page overflow, clipped text and scrolling containers: no findings. Search results and Details were checked by eye at 1280 and 390 in both themes.

## "May drop" flag

Server side (`add_drop_fields` in `src/app.py`, no extra model calls): a flight may drop when the lowest predicted price among the curve points at or after today's days-before-departure, plus the configured horizon prediction, is at least `flag_threshold_pct` below the listed price. The search response gains `may_drop`, `drop_pct`, `drop_best_days_left` and `drop_price`.

UI:
- A quiet line at the foot of the ticket's main part: a small down-arrow, the words "May drop" and "about 12% by 14 days out", set in the "down" green, separated by a dotted rule. A thin green line runs along the top of the stub. It is distinct from the red up-arrow and "Large rise" label, and carries meaning in words and arrow shape, not colour alone.
- Filter "Price may drop (n)" beside the airline and stops filters. The count follows the airline and stops filters, so it always equals the number of results when ticked. It works with sorting and the pager (page resets to 1), and stays ticked across a new search. If a route has no drops the empty state says so and offers "Show all flights".
- Sort option "Biggest expected drop".
- In Details, the lowest point is ringed on the curve with a direct label, and one sentence states when and how much, ending "It is an estimate, not a promise."
- The footer states that a "may drop" flag is an estimate, not a promise.

## "Big save" sticker

A may-drop flight is a big save when its expected fall is at least `big_save_threshold_pct` (default 20, range 5 to 80) or its expected rupee saving is at least `big_save_min_inr` (default 1500, range 0 to 50000; 0 turns the rupee test off). Both settings live in `DEFAULTS`/`validate_config`, in `GET/POST /api/config` (reset restores them) and in the Settings tab. The search response adds `big_save` and `save_inr` (0 when not a may-drop flight).

UI: the sticker replaces the quieter "May drop" line on big-save tickets and keeps its detail text ("about 57% by 21 days out"). It is a filled badge in the "meaning" green with a dashed outer outline, a slight tilt, a hand-drawn price-tag-with-down-arrow icon, "BIG SAVE" in heavy capitals and "Save about ₹3,409". It has an `aria-label` such as "Big save: about 22 percent, 1,840 rupees cheaper". Text on the badge uses `--on-ink` over `--on-bg` (6.3:1 in light, white on `#2B6B4E`; 8.6:1 in dark, `#10140F` on `#86BD94`; both pass AA), and the words and icon carry the meaning as well as the colour. A "Big saves (n)" filter sits beside "Price may drop (n)", with the same count rule (counts follow the airline and stops filters), the pager and sorting. "Biggest expected drop" puts big saves first, then orders by percentage. The Details heading carries a small "Big save" tag and the sentence adds the rupee saving.

## United States market

**Switch.** A two-part segmented control above the search card: "India (INR)" and "United States (USD)". The choice is remembered per visitor in `localStorage` (`fs-market`, in try/catch). India is the default and its behaviour, endpoints and look are unchanged.

**Backend (`src/app.py`).** `GET /api/us/meta` and `POST /api/us/search` delegate to `src/us_service.py`, imported lazily on first use, so the app starts and India works when the US files are missing. Failures return `503 {"available": false, "error": "US data is not installed on this server."}` and are retried at most every 20 seconds. Bad input returns a JSON `400` (missing fields, same airport twice, bad body) or `404` (unknown route or date format, from the service); an out-of-range date returns `200` with an empty list and the service's note. Each search response includes `elapsed_ms`. The "may drop" and "big save" percentage thresholds come from the shared config; the US dollar amount uses the service default. JSON key order is now preserved (`app.json.sort_keys = False`) so the feature families stay in model order.

**Search card in US mode.** Origins and destinations come from `us_meta()`, and the destination list only offers valid pairs for the chosen origin. The date picker is limited to the searchable range. The class selector is replaced by a read-only "Listed as of" field, since cabin is not in the data and the prices are a snapshot. The results show the snapshot date in the facts line and in a note, and an honest sentence that the model improves only slightly on "the price stays the same".

**Results.** The same list, pager, filters, sorting, Approve, "May drop" and "Big save" tags and Details are reused through an adapter that maps a US flight to the India shape. Money is formatted through one function (`$1,234` or `$85.18`). Times are real clock times with a +1 marker for next-day arrivals. "Parameters the model used" is grouped by feature family, each with its column count, as collapsible sections (the first two open).

**Model & Insights, "United States" tab.** Honest note, key counts, the headline accuracy next to the two naive baselines, the inputs by family with column counts, accuracy by horizon, data sources and licence notes (Open-Meteo is non-commercial) and data limits. Tables the service returns in unknown shapes go through a generic renderer. If the service is missing the tab shows the "not installed" state.

**Mock (kept, gated, unused in normal runs).** `src/us_mock.py` mirrors the `us_service` output shape and is used only when `FARESIGHT_MOCK=1` *and* the real service cannot load. With the real service installed it is never imported. Its responses carry `"mock": true` and the UI marks them "Sample data".

**Verified against the real service** (`us_service`, snapshot listed as of 2026-09-30, departures 2026-10-07 to 2027-03-29):
- 50 routes, 14 airlines, 197,189 flights in the snapshot, **78** model inputs in six families (basics 11, price history 22, route market 7, holidays and calendar 22, fuel 6, weather 10). The brief mentioned 84 to 88; the service reports 78.
- Searches took about 20 to 60 ms server-side and about 0.5 s end to end in the browser (the first request after a restart loads the model and takes longer; not timed).
- Most flights show predicted close to listed, as the model note says. Drops are rare: of 51 sampled route and date combinations (17 routes, 3 dates) the most was 4 may-drop flights (ORD to BOS, 15 Nov 2026, all four also big saves); ORD to BOS on 20 Jan 2027 has two plain "May drop" flights; ATL to DFW on 15 Oct 2026 has none, which shows the friendly empty state when the filter is on.
- Out-of-range dates return an empty list plus a note (shown as "No flights on that date"); same airport, unknown pair, bad date and non-numeric horizon return JSON errors.
- **Bug found and fixed:** some inputs are missing (for example price history on a flight's first quote). `us_service` returned them as NaN, and Python writes a bare `NaN` that browsers cannot parse, so those searches showed "We could not reach the server". `src/app.py` now converts NaN and infinity to null in the US responses (`finite_json`), and the parameter list shows "not available". The cause is in `us_service._fmt`, which checks `isinstance(v, float)` and so misses numpy float32 values; the service owner may want to change that to `np.floating`.
- Insights, "United States": real headline accuracy (model 93.3% against 93.0% for "price stays the same" and 66.3% for the route median), the family table, a compact horizon table (the model is only clearly better than the last price from 5 days ahead, by 0.3 to 3.7 points), and sources with licences.

## May drop, revised (shared rule for India and the US)

**Why.** The first definition compared the lowest curve point with the listed fare. The listed fare is a scraped last-minute fare, the curve is model output, and a single odd model point could trigger a tag. That tagged flights whose own card showed a price rise.

**Rule (`src/drop_logic.py`, used by `/api/search` and, as a post-processing step, `/api/us/search`; `src/us_service.py` is untouched).**
1. Only the future counts: curve checkpoints with fewer days before departure than today, plus the model's own price for today (`p0`). For India this is one extra batch prediction per search at today's days-before-departure (clamped to the model range); for the US the first point of the service's curve is taken as `p0`.
2. The expected low is the lowest level held at two adjacent future checkpoints, `min over pairs of max(p_i, p_i+1)`. A one-point dip cannot produce it.
3. A flight may drop only if that low is at least `flag_threshold_pct` below both the listed fare and `p0`. `drop_pct` and the saving are measured against the listed fare, from the rounded prices the card shows.
4. Big save keeps its thresholds.
5. A tagged card shows the expected low as its FareSight figure ("Expected low", with the days out) and the change against the listed price, so figure, arrow and tag agree. Untagged cards keep the horizon-based figure. New API fields: `p0`, `drop_pair_days_left`, `card_price`, `card_kind`, `card_days_left`, `card_change_pct`.

**UI.** Details draws points before today faintly and dashed (not used), starts the solid line at today's model price (open ring), highlights the two checkpoints that hold the low with a green segment and ring, and labels the low in the margin. The sentence gives the level, the days out, the percentage below listed, the saving for big saves and the model's price for today, and always ends "It is an estimate, not a promise." The filter tooltips, empty states and a "How the May drop tag works" note on the Settings tab use the same wording. Empty state: "The model does not expect a sustained fall on this route and date; most fares are expected to rise toward departure."

**Tests.** `src/tests_drop_logic.py` (run from `src/` with the project venv): unit tests (one-point dip, sustained dip, dip before today, both-comparisons rule, two-points requirement, horizon point, big-save thresholds) and an invariant check over many real searches on both markets: a tagged flight's card shows its expected low below the listed price with a negative change, tag percentage equals the card change, the low is below `p0` by the threshold, untagged flights keep the horizon price, big save implies may drop.

## Prediction engine switch (India)

**What.** Settings, top card "Prediction engine": a table of engines with a radio, plain-English description, size, speed, load time and the three accuracy scores (same period, recalibrated, unseen period) with within-10% and average miss. The active engine is tagged, the best score per scenario is tagged "best", Random Forest is tagged "Requested by the portfolio manager" and its unseen-period weakness is stated plainly next to it, and an engine that cannot load on this server is listed greyed out with the reason. "Use this engine" shows a loading state (the small plane on a dashed line), then a toast ("Now predicting with XGBoost (ready in 0.6 s). Search again to see new prices."). The card says that all engines share the same training data, that they differ by tenths of a point on known periods and by several on an unseen one, and that **the US market uses its own price-history model**.

**API.** `GET /api/engines` returns the manifest (from `src/engines.py`), which engine is active and, per engine, `available` / `unavailable_reason`; a missing file or library is a disabled row, never an error. `POST /api/config {"algorithm": key}` switches (merging into the saved settings, warm-loading the engine and returning `engine.load_seconds`; an engine that cannot load returns `409` with the reason and nothing is saved). A settings save that omits `algorithm` keeps the current engine; an unknown key falls back to `hgb`. `algorithm` is in the defaults, `GET /api/config` and the reset action. The deployed model (`hgb`) is always the app's own `MODEL`, so it is never loaded twice.

**Search.** `predict()` and the curve use the active engine (`engine.predict(feats)` returns the log price, so nothing downstream changed). The response gains `engine: {key, label, requested, fallback, mock}`; if the chosen engine fails at search time the search falls back to the deployed model and says so in `engine.fallback`.

**Demo page.** The facts line shows "Engine: XGBoost", Details says "Engine: XGBoost." above the inputs, and when the tab regains focus after the engine was changed elsewhere a note says "The prediction engine is now X; these prices were made with Y. Search again", with a button that re-runs the search.

**Mock.** `src/engines_mock.py` is used only with `FARESIGHT_MOCK=1` and fewer than two real engines listed (sample engines that bend the deployed model's log price; marked "Sample data"). Screenshots made with it have `MOCK` in the file name.

**Tests.** `tests_drop_logic.py` repeats the tag, card and filter invariants for every loadable engine (the engine is set in memory, nothing is written).

## QA round 1 fixes (docs/QA_REPORT_1.md)

**One story on the card.** The big slot is always the FareSight price at the chosen horizon, with its change against Listed. A May drop adds a second green line, "Expected low", with its own price, percentage and timing. The "Large drop / Large rise" labels and edge colours are gone: the only tags are May drop and Big save. The rule now also requires the big FareSight price to be at least 1% below Listed, so every tagged card visibly falls in both places and the filters can never show a rise. Tag, expected-low line and API use the same rounded prices (`drop_pct` is computed from the shown prices; the UI rounds half away from zero everywhere).

**Plain definitions on screen.** A permanent legend line (Listed, FareSight), a "What do these labels mean?" disclosure (Listed, FareSight, May drop, Big save, FareSight's estimate for today, Approve, hidden airlines, "estimates, not promises"), the basis line ("Based on fares scraped in 2022 and 2023, not live prices or current schedules"; US: "Snapshot as of 30 Sep 2026") and one accuracy sentence with the three held-out scores (from `models/metrics.json`) and the limit that each flight's price was observed once. The jargon chips are replaced by sentences. Sorts are renamed: Listed price, FareSight price, Biggest fall vs listed, Biggest rise vs listed, Biggest expected later drop, Airline.

**Airlines that no longer operate.** GO FIRST, Vistara and AirAsia (India) are hidden by default and marked "Historical data, airline no longer operates" when shown. Settings has "Include airlines that no longer operate" (default off). The server filters them before counts, tags, filters and pagers; the response reports `hidden_defunct`. Other airlines in the data (Air India, IndiGo, SpiceJet, Akasa Air, Alliance Air, Star Air) operate today.

**Dates.** India: earliest is tomorrow, and the note beside the field says dates beyond 50 days ahead are priced as 50 days out. US: the valid window is printed under the form and the default is 30 days after the snapshot, clamped into the window.

**Approve.** "Shortlist only. Nothing is booked." sits next to the button, with a running count and "Clear approvals". Approvals persist across searches on the same market (kept per flight in `sessionStorage`, so they survive a refresh and end when the tab closes), "Approve all" is hidden on empty pages, and the legend says what Approved does.

**US honesty.** US tags are expressed as "N days after the listing date" (all within the 14 tested days); the chart axis for US counts days after the listing date; the horizon table derives one verdict from both scores (a mixed result is called mixed) instead of repeating the service's wording.

**Navigation and robustness.** Market, route, date, class, filters, sort and page are in the query string; Back, Refresh and sharing work. The market comes from the URL first, then `localStorage`, and storage is only written when the user clicks a market (a page load never changes it). A new search resets the filters; the count line says when filters are on. Network errors retry once automatically, then show "We couldn't reach the service. Please try again in a moment." with a Try again button; same-city India searches are caught before the request.

**Speed.** The search made one model call per day on the curve, each with a large fixed cost (about 0.4 s). It now makes one call for all days, and caches the model output per route, date, class and engine, so repeats and threshold changes are instant. Delhi to Mumbai: about 4 s before, about 1.2 s first time, 0.01 s repeated.

**Numbers and wording.** One rounding rule for every score (within ±10% reads 77.6% on every tab), training time stated once as 6 to 7 minutes (6.6 in the benchmark run), "Goal: 95%", and plain wording for MAPE, "target", importance and the hyper-parameters. Raw names (te_flight and friends) are translated; the technical tables sit under a collapsed "Technical details". US detail rows show names instead of internal codes and weekday numbers.

**Polish.** Tap targets are at least 40 px, keyboard focus returns to the pager and to the card button after a click, the hero plane flies in the right half only, the tab bar is opaque with a shadow, and time windows are labelled "(time window)" wherever an exact time is missing.

## XGBoost as the default engine

`DEFAULTS["algorithm"]` is now `xgboost` (also what Reset restores). If that engine's file or library is missing on a server, `/api/engines` reports `fallback` (wanted engine and reason), the engine card shows "XGBoost is the default but is not available on this server (reason). Predictions are using HistGradientBoosting instead.", and the search page shows the same sentence with the engine named in the results line; no error is raised and the scores shown are the ones of the engine that actually made the prices. The default engine is loaded in the background at start-up so the first search is not slow.

The Model & Insights page follows the active engine: Metrics shows its three scores, within-10% and average miss (from `models/algorithms.json` through the manifest); the Model tab shows its name, family, description, size and settings; the algorithm comparison marks it "Active". The history of the first model is kept and labelled: "How the first model (HistGradientBoosting) got from 86.9% to 93.0%", and the importance chart says it was measured on the first model, because no importance file exists for XGBoost. The search page's accuracy sentence uses the engine that made the prices (XGBoost: 93.4 / 86.9 / 67.7).

Checked with the real XGBoost engine: switching, searches (Delhi to Mumbai in 0.4 s), the fallback with LightGBM forced unavailable, and the tag, card and filter invariants for hgb and xgboost.

## QA round 2 fixes (docs/QA_REPORT_2.md)

- **One active marker (N-01, N-04, N-05).** No label says "current" any more. The first model is "First model (HistGradientBoosting)" and tagged "Reliable"; XGBoost is "Recommended"; only the active engine carries "Active". Speed, size and scores for every engine come from the engine manifest on every tab (the algorithm comparison overrides its own speed and size figures with the manifest's). The score write-up follows the active engine, and sentences about the first model say so. Technical details has plain meanings for XGBoost's settings, friendly input names, and no file names, shell commands or raw column names.
- **Engine selector (N-03, N-06, N-07).** Recommended / Reliable / Also evaluated tags; Ridge, Neural net, Random Forest and Extra Trees carry "Included for comparison. Less accurate on this data; prices may look unusual." and need a confirmation ("Switch anyway") before they can be activated. A chip at the top of the page names the active engine, says it applies to everyone using the server, and offers "Back to recommended (XGBoost)"; the search page shows the same chip. Uninstalled engines read "Not installed on this server", with the reason in a collapsed technical detail. The "requested by the portfolio manager" note moved into Random Forest's collapsed technical detail. The engine table no longer traps scrolling, and /config shows the first tab at once.
- **Big save needs a real percentage (N-10).** A big save is a May drop whose fall is at least the big-save percentage, or a large amount that is also at least a 10% fall; an amount alone never qualifies (`big_save_rule` in `drop_logic.py`, used by both markets). Settings text, glossary and tests are updated; the invariant test checks every big save is at least a 10% fall.
- **Expected low matches the chart (N-11).** The chart draws the thin model line plus a bold smoothed line: the level held at two dates in a row. The expected low is that line's lowest point, and its date is the later date of the pair (`drop_best_days_left`). When the very last date dips lower, the Details sentence says it is not counted as lasting.
- **Wording (N-12, N-08).** The hero reads "Listed fare, predicted for departure"; the card says "historical fare" (India) or "as of 30 Sep 2026" (US). An unsupported US route in the URL shows "No data for that route"; the US out-of-range message uses human dates.
- **Repeated Listed fares (N-02).** Equal sort keys fall back to departure time (exact when known, else the middle of the window) then flight code, so the order is stable. A "same fare as N other flights" hint shows when three or more flights of an airline share a Listed fare, and the legend says some airlines have one typical fare for all their flights on a route. Nothing is invented.
- **Shortlist and search feedback.** "Show shortlist" filters to approved flights and the count says how many are in this search. Searching scrolls the results into view with the loading indicator.

## Beginner layer: Price Lab, How it works, Start here

Audience: someone non-technical who is weak in maths. Plain words, short sentences, concrete numbers, nothing claimed that the reports in `docs/` do not support.

**Price Lab (`/lab`).** One table with the 18 inputs the engine receives: Input, Your value (editable: selects, numbers, sliders with limits from the training data), What it means, Effect on price. Under it: the live predicted price (about 300 ms after the last edit), the original price of the loaded flight, the difference, a "Biggest change" helper (which single edit moved the price most, measured one input at a time) and a waterfall from the base fare to the final price. The three "typical price / supply" inputs and distance are marked "Calculated from past data; edit to see how strongly the model relies on it". Start from a typical flight, from any flight found by route and date, or from "Open in Price Lab" on a card's Details.
- `POST /api/whatif {values, original?, engine?}` builds one encoded row through the same encoder the search uses (typical prices, supply and distance are set as given), predicts with the active (or chosen) engine, and returns `steps` (one per input) whose effects add up to `price - base_price`. Values are validated: unknown choices, non-numbers and NaN return a clean 400 message; numbers outside the training range are clamped and reported in `warnings`.
- Breakdown method: XGBoost uses exact TreeSHAP; the other engines use sampled Shapley values (20 random orders of switching each input from a typical flight to this one, all predicted in one batch; every order adds up exactly). The page says which one it used. `src/lab.py` holds the pure parts. `engines.explain()` was not available when this was built, so the explanation is implemented here, not taken from `engines.py`.
- `GET /api/lab/spec` returns the input descriptions and the typical-flight reference.

**How it works (`/how`).** A guided tour: what FareSight does, where everything is (real paths and the GitHub repo), where the data came from and whether it is real, the 18 inputs with their importance, fuel (honest answer: not in the India model, about 0.2 points at most in the US model), how the model learns (five steps and two small diagrams), the accuracy score with a hands-on calculator (seeded with 2000 and 2050), the algorithms (one card each, scores from the engine manifest), what Approve is, and client questions. `GET /api/facts` supplies the numbers (metrics, importance, engine scores, row counts) from the `models/` files.
- Importance is shown as accuracy points lost when an input is scrambled (about 71, 20, 11, 11, others under 5). These are not percentages and do not sum to 100; the earlier "about 90% / 62%" share wording was removed after the lead's correction.

**Model & Insights.** The first tab is "Start here" (what it is, the three scores in one sentence each for the active engine, what it was trained on, links). Tabs are grouped: Overview (Start here, Settings and engine) and For engineers (Metrics, How we got the score, Model & parameters, Data & sources, Algorithm comparison, United States). Every tab has a one-line subtitle under the tabs, every block heading has one, and metric names carry a "?" that opens a plain sentence on tap or click (accuracy, average percentage error, within ±10%, average miss, rounds, learning rate, importance, the three tests and more).


## Algorithm walkthrough: "How the price is calculated" (`/walkthrough`)

Nav item "How the price is calculated", linked from Price Lab, How it works and the Start here tab.

**Part 1, the toy (browser only, no server call).** Eight editable made-up flights (up to 12, add/delete/reset), a new flight, a learning-rate dial (0.1 to 1) and a trees dial (1 to 6). A tiny gradient-boosting routine with one-question trees: start = average price; each round computes leftover errors, tries every question, picks the smallest remaining squared error, and writes out every sum. Shows the questions tried (chosen row highlighted), a running table of each flight's guess with the total miss per round, the pricing table for the new flight with a one-line equation, a "Check the score" box (same sum as How it works), and two simplified comparisons (random forest on seeded random subsets, straight line by least squares). `src/toy.py` is a Python mirror, pinned by `test_toy_reproduces_hand_calculation`.

**Rules that reproduce the hand calculation** (5,800 → 4,933 → 5,608 → 5,042; steps -867, +675, -567): questions in tie-break order Business, "under N days" for N = 10, 20, 30 (round numbers only), 1 stop, 0 stops; questions that split the flights identically count once; a question is used at most once. The brief said "under N for every distinct value" but that makes "under 5 days" beat "under 10 days" in round 2, and an unrestricted search asks "Is it Business?" again in round 2, so the brief's hand calculation was not the unrestricted minimum. Round numbers plus use-once reproduce it exactly. Squared errors are shown in millions, not rupees.

**Part 2, the real model (server).** `POST /api/trace` takes the `/api/whatif` body, returns `engines.trace()` (start value, cumulative price after k trees, decision paths for the first trees and the three most influential, `questions_used`, `summary_sentence`) plus the normal breakdown under `breakdown`. Other engines get `available: false` with a clean message and the breakdown only. The page carries the Price Lab flight over via `localStorage['fs-lab-values']` (try/catch), or starts from a typical flight; edits recalculate after 300 ms. Sections: inputs, start price, chart of price after k trees (evenly spaced ticks, labelled as not to scale), tree paths as vertical lists (log-price thresholds shown as "about ₹..."), questions asked most, the addition with "and N more small steps", and a toy-vs-real table (3 vs 3,500 trees, 0.5 vs 0.05, up to 511 end points, 8 vs 745,375 flights; 511 and 0.05 read from the loaded model's parameters).
