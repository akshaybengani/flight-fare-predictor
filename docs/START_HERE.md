# FareSight: start here (plain-English guide)

Written for someone new to tech and to maths. No formulas you cannot do on a phone calculator. Read it top to bottom once; then use the "If someone asks you..." section as a cheat sheet.

---

## 1. What is FareSight, in two sentences

FareSight shows a flight's **listed price** next to the price a computer model **expects** for the same flight, so a pricing person can see where prices may rise or fall. It is a **demo (proof of concept)**: it shows the idea works, it is not a live booking product.

The original brief (from the project call) was: "a MakeMyTrip-style site where we see the original price and the predicted price with an Approve button".

---

## 2. Where is everything?

| What | Where |
|---|---|
| The whole project folder | `/Users/akshay.bengani/Work/flight-poc` |
| Source code online (private) | https://github.com/akshaybengani/flight-fare-predictor |
| The website (what you demo) | runs from `src/app.py`, pages are in `templates/`, styling in `static/` |
| How the model is trained | `src/train.py` (first model), `src/engines.py` and `src/build_engines.py` (the 8 deployable models) |
| How the data was cleaned and merged | `src/build_dataset.py`, `src/features.py` |
| The "May drop / Big save" rule | `src/drop_logic.py` |
| The US market | `src/us_service.py` |
| The datasets | `data/` folder (not in GitHub: too big and not ours to share) |
| The trained models | `models/` folder (`fare_model.joblib` first model; `engines/xgboost.joblib` the default) |
| Written reports | `docs/` folder (data scout, enrichment, algorithm comparison, QA reports, this guide) |
| Start the website | `cd /Users/akshay.bengani/Work/flight-poc/src && ../.venv/bin/python app.py`, then open http://localhost:5055 |
| Copy for the home server | the Samba share `bulk`, folder `faresight` (model and data files) |

Think of it as: **data** (past prices) goes into **training** (a computer learns patterns) to make a **model** (a file), and the **website** asks the model for a price.

---

## 3. What data did we use, and is it real?

**India (the main demo)**
- Two datasets from Kaggle (a website where people share data): fares scraped (copied automatically) from the **EaseMyTrip** booking site, one set from early **2022** and one from early **2023**.
  - Links: https://www.kaggle.com/datasets/rohitgrewal/airlines-flights-data (2022) and https://www.kaggle.com/datasets/yashdharme36/airfare-ml-predicting-flight-fares (2023).
- Together: **745,375 prices**, 7 cities (Ahmedabad, Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai), 9 airlines, 2,244 different flights.

**United States (the second market)**
- Kaggle "Airfare Time Machine": 38.6 million price quotes for 50 US routes, collected over 2026. https://www.kaggle.com/datasets/pooyanayebi/airfare-time-machine-us-flight-prices

**Is it real or random numbers?**
- It is **real scraped data**. Nobody on our team made up fares. We added **no synthetic (fake) data anywhere**.
- We cannot *prove* it is genuine, because there is no official answer sheet. We checked that it behaves like real fares: longer flights cost more, connecting flights differ, fares jump as departure gets close, and the two files line up row by row.
- Two other downloaded datasets looked fake (Bangladesh fares and an "oil vs airfare" file) and we **did not use them**.
- What it is **not**: a live feed from the airlines. It is what a booking site showed on the day it was scraped. India's data is from 2022 and 2023, so some airlines in it (GO FIRST, Vistara) no longer fly as before. The app hides them by default.

**Why we say "scraped"**: a program visited the booking site many times and saved the prices it saw.

---

## 4. What the model looks at (the "features")

A **feature** is just one fact about a flight that the model is allowed to look at. The live India model uses **18**:

| # | Feature | Plain meaning |
|---|---|---|
| 1 | Airline | IndiGo, Air India, SpiceJet... |
| 2 | From city | where it starts |
| 3 | To city | where it lands |
| 4 | Class | Economy or Business |
| 5 | Stops | 0, 1 or 2+ |
| 6 | Duration | hours in the air |
| 7 | Distance | kilometres between the cities |
| 8 | Departure window | morning, afternoon, evening, night... |
| 9 | Arrival window | same idea |
| 10 | Departure hour | exact hour (only some data has it) |
| 11 | Arrival hour | exact hour (only some data has it) |
| 12 | Weekday | Monday to Sunday |
| 13 | Month | which month of the year |
| 14 | Days before departure | how many days ahead you look |
| 15 | Which data set | 2022 or 2023 (prices were higher in 2023) |
| 16 | Typical price of this flight | the average past price of this flight number and class |
| 17 | Typical price of this airline on this route | the average past price for that airline and route |
| 18 | Flights on the route | how many flights exist on that route |

**Which ones matter most?** Numbers 16 and 17 matter far more than everything else. In our test, hiding #16 (typical price of the flight) made the score fall by about 71 points, and hiding #17 by about 20 points. Days before departure and duration each cost about 11 points. All other features cost under 5 points each. (These are "points lost when the information is scrambled", not percentages that add up to 100.)

Plain reading: **the model mostly says "this flight usually costs about X" and then adjusts a little for days left, duration, time of day and so on.** That is honest, and it is also why it is good on flights it has seen and weaker on brand-new situations.

The US model uses about 88 columns, because the US data shows the same flight's price on many days (its own recent price history is the strongest clue).

---

## 5. Does fuel price change the price?

**In the real world: yes, but slowly.** Fuel is roughly a third of an airline's costs. When fuel gets expensive, airlines raise base fares or add a "fuel surcharge", usually with a delay of weeks.

**In our model: the live India model does not use fuel price at all.** We tested adding fuel and exchange-rate columns (from FRED, a public US data site) and the score did not improve in any way we could trust. The reason is simple: our India data is just two snapshots, one from 2022 and one from 2023. Fuel was different in those two years, but so was everything else (demand, airlines, season), so the model cannot tell what fuel did. In the US model fuel is included; dropping it costs only about 0.2 points, which is too small to say fuel caused anything.

**What to say:** "Fuel is a known real-world driver. We tested it, but our data cannot separate its effect yet. To measure it you need daily prices over a long period."

The same is true for weather, holidays and festivals: we added them, and they did not help measurably on this data.

---

## 6. How a model learns (no maths)

1. We show the computer **hundreds of thousands of past flights** with their facts (the features) and their **real price**.
2. It looks for patterns: "Flights like this one usually cost about this much."
3. We **hide** a part of the data (it never sees these flights or prices while learning).
4. We show it only the facts of those hidden flights and ask for a price.
5. We compare its answers with the real prices. That comparison is the **accuracy score**.

Once it is trained, the website just asks it: "Here are the facts of this flight, what price do you expect?"

**Rough "equation" idea:** the model's price is roughly a typical fare, then pushed up or down by each fact. For example: typical fare ₹4,000, +10% because the departure is close, −5% because it is a Tuesday, +8% because of a long duration. The **Price Lab** page in the app (if it has been built) lets you change one fact and watch the price move.

---

## 7. The accuracy score, step by step

It is **not true or false.** Each flight gets a score between 0 and 100, and we average them.

**Example:** real price ₹2,000, model says ₹2,050.
1. Difference: 2,050 − 2,000 = **₹50** too high.
2. As a share of the real price: 50 ÷ 2,000 = 0.025 = **2.5%** miss.
3. Score for this flight: 100 − 2.5 = **97.5%**.

If the model had said ₹1,000 instead: the miss is 1,000 ÷ 2,000 = 50%, so the score is **50%**. If it says ₹2,000 exactly, the score is 100%.

**The overall number** (like 93.0%) is the average of that miss over tens of thousands of hidden flights, subtracted from 100. So "93.0%" means: **on average the model is off by about 7%**. Technical name: MAPE (mean absolute percentage error); accuracy = 100 − MAPE.

**It does NOT mean** "93 out of 100 flights are right". For that there is a second number, **"within 10%"**: the share of flights whose predicted price is no more than 10% away from the real price. This one *is* yes or no per flight: ₹2,050 for ₹2,000 is within 10%, so it counts as a yes. Our model gets about **78 out of 100** flights within 10%.

A third number is the **average miss in rupees** (about ₹1,500 in the best test).

**Why three scores (93, 87, 66)?** They are three exams of different difficulty:
- **93%: same period.** Hidden flights come from the same weeks as the training flights, so the model has seen the same flights on other days. Easiest exam.
- **87%: small recent sample.** Trained on 2022 plus a little of 2023, tested on the rest of 2023.
- **66%: unseen period.** Trained on 2022 only, tested on 2023: a whole new year with different price levels. Hardest, and the honest test of "can it predict something new".

**The 95% target** from the call is not reached. The main reason is the data: each flight's price was seen only once, so the model never saw a price change over time.

---

## 8. The algorithms, one by one

An **algorithm** is just the method the computer uses to find patterns. We tried eight (plus a simple baseline). Scores below are the three exams (same period / small recent sample / unseen period).

**0. Median lookup (the baseline).** For each airline, route and class, remember the typical (middle) price and repeat it. No learning. 75 / 67 / 62. Every real model must beat this.

**1. Ridge (a straight-line formula).** Gives each fact a weight and adds them up, like price = 500 + 3 × duration + 700 × business. Fast, simple, easy to explain, but prices do not behave like straight lines. 81 / 72 / 66. (Surprisingly good on the hardest exam because it is simple.)

**2. The "decision tree" idea (a flowchart of questions).** "Is it Business? Yes. More than 1 stop? No. Departs evening? Yes → about ₹9,000." A tree is a flowchart of yes/no questions that ends in a price. All the next algorithms are built from many trees.

**3. Random Forest (many trees voting).** Build hundreds of different trees, each looking at a random part of the data, and **average** their answers, like asking a crowd. Steady and hard to fool, but large files. 92 / 87 / 44. It does badly when the year changes. (Requested by the portfolio manager.)

**4. Extra Trees (a random forest with even more randomness).** Same idea, trees split at random points; faster. 93 / 87 / 57.

**5. Gradient boosting (trees that fix each other's mistakes).** Build one small tree, see where it is wrong, build the next tree to fix those mistakes, and repeat thousands of times. Like a student doing practice exams and focusing on past errors. This family is usually the best for tables of numbers, and it is what we use. Four versions:
 - **HistGradientBoosting:** comes built into scikit-learn (a common free Python library). Our first model. 93.0 / 87.0 / 66.2.
 - **XGBoost:** a very popular, carefully optimised version. **The default now.** 93.4 / 86.9 / 67.7.
 - **LightGBM:** another popular version, faster training. 93.2 / 87.2 / 60.9.
 - **CatBoost:** designed to handle category columns (like airline names) neatly. 91.5 / 86.3 / 67.4.

**6. Neural net (MLP: layers of tiny calculators).** Inspired loosely by the brain; layers of simple calculations. Great for images and text, usually not better than trees for tables like ours. 87 / 80 / 47. Included for comparison.

**Why XGBoost if it is best, but we started with HistGradientBoosting?**
- HistGradientBoosting was the first model because it needs nothing extra to install.
- Later we tested eight algorithms fairly. XGBoost came out ahead in two of the three exams, by only a few tenths of a point (93.4 vs 93.0; 67.7 vs 66.2), and slightly behind on one (86.9 vs 87.0).
- They are cousins (both boosted trees), so the gap is small. We switched the default to XGBoost, and HistGradientBoosting is the automatic backup if XGBoost is missing on a server.
- Honest summary: **"The top boosted-tree models are roughly equal; XGBoost is marginally ahead."**

---

## 9. What is "Approve", and why is it there?

It comes from the original brief: the screen shows the **original (listed) price and the predicted price**, and a person (a pricing or revenue manager) **approves** the predicted price. In a real system, approving would send that price to the booking platform and keep a record of who approved it.

In **this demo**, Approve only **marks the flight on the page** (the listed price is crossed out and the predicted price turns green). **Nothing is booked, saved or sent anywhere**, and it resets on refresh. It exists to show where a human decision would fit.

---

## 10. What do the tags on flights mean?

- **May drop:** the model expects a lower price some days from now (at least 10% lower than the listed price, and the dip must hold for more than one point so it is not a blip). The card shows the expected low price.
- **BIG SAVE:** a "may drop" flight where the fall is large (20% or more, or a big rupee amount together with a real percentage drop).
- They are **estimates, not promises**. The model mostly predicts prices rising close to departure, so many routes show few or none.

---

## 11. The Model & Insights page, tab by tab

| Tab | Plain meaning |
|---|---|
| Settings | Dials for the demo: how much change to flag, how many days ahead, which engine (algorithm) is active, whether to show airlines that no longer operate |
| Metrics | The three exam scores for the active model |
| How we got the score | The story of how the first model improved, step by step |
| Model & parameters | The technical settings of the first model (for engineers) |
| Data & sources | Which datasets, where from, what was cleaned |
| Algorithm comparison | The "bake-off": all algorithms on the same exams |
| Engine | Choose which algorithm powers the prices |

---

## 12. If someone asks you...

| Question | Short answer |
|---|---|
| Where is the code? | In `/Users/akshay.bengani/Work/flight-poc` and the private GitHub repo `akshaybengani/flight-fare-predictor`. |
| What data is it? | Real fares scraped from EaseMyTrip in 2022 and 2023 (India, 745k prices) and a 2026 US dataset (38.6M quotes), both from Kaggle. |
| Is it real or random? | Real scraped fares. We made nothing up. Not live airline data. |
| Where did you get it? | Kaggle, community-uploaded. Links in section 3. |
| What does the model use? | 18 facts about each flight (airline, route, class, stops, duration, time, weekday, days left, typical price...). |
| Does fuel/weather/holiday matter? | We tested them; they added nothing measurable on this data. Real-world fuel does matter, but our data cannot separate it. |
| How accurate is it? | On average about 7% off on flights from the same period (93%); 87% with a small recent sample; 66% on a new year. The 95% target is not reached. |
| Which algorithm? | XGBoost (boosted trees), tested against seven others. |
| What is Approve? | A demo shortlist button showing where a pricing manager would approve the predicted price. Nothing is booked. |
| What would make it better? | Daily price capture for the same flights, seat availability, a longer history or a licensed fare feed. |
| Is it live? | No. Historical scraped fares. |

---

## 13. Words you may hear

- **Dataset:** a big table of examples. **Row:** one example (one flight price).
- **Feature:** one fact the model may look at. **Target:** the thing to predict (the price).
- **Train / test:** learn from some rows; get graded on rows it has never seen.
- **Model:** the file the computer produces after learning.
- **Scrape:** copy data automatically from a website.
- **Snapshot:** prices seen on one day.
- **MAPE:** average percentage miss. **Accuracy here:** 100 − MAPE.
- **Overfitting:** memorising the past instead of learning patterns (scores great on old data, badly on new).
- **Baseline:** a simple method every real model must beat.
