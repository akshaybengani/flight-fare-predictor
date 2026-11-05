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
