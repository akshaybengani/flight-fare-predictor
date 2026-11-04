"""Merge the real fare datasets into one training table.

period 2022: airlines-flights  (no dates; days_left only. Scrape date assumed
             2022-02-11 per the published description of the EaseMyTrip scrape)
period 2023: airfare-ml        (booked 2023-01-15, journey dates known)
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "data" / "merged.parquet"

AIRLINE = {"Air_India": "Air India", "GO_FIRST": "GO FIRST", "Indigo": "IndiGo",
           "AkasaAir": "Akasa Air", "AllianceAir": "Alliance Air", "StarAir": "Star Air"}
STOPS = {"zero": 0, "non-stop": 0, "one": 1, "1-stop": 1, "two_or_more": 2, "2+-stop": 2}
BUCKET_2022 = {"Early_Morning": "Before 6 AM", "Late_Night": "Before 6 AM",
               "Morning": "6 AM - 12 PM", "Afternoon": "12 PM - 6 PM",
               "Evening": "After 6 PM", "Night": "After 6 PM"}


def _hour(col):
    """'20:30\nDelhi' -> 20.5"""
    t = col.str.split("\n").str[0].str.split(":", expand=True).astype(float)
    return t[0] + t[1] / 60


def load_2022():
    d = pd.read_csv(DATA / "airlines-flights/airlines_flights_data.csv").drop(columns="index")
    d = d.drop_duplicates()
    scrape = pd.Timestamp("2022-02-11")
    return pd.DataFrame({
        "period": 2022,
        "airline": d.airline.replace(AIRLINE),
        "flight_code": d.flight,
        "source": d.source_city, "destination": d.destination_city,
        "class": d["class"], "stops": d.stops.map(STOPS),
        "dep_bucket": d.departure_time.map(BUCKET_2022),
        "arr_bucket": d.arrival_time.map(BUCKET_2022),
        "duration_h": d.duration, "days_left": d.days_left,
        "journey_date": scrape + pd.to_timedelta(d.days_left, unit="D"),
        "price": d.price,
    })


def load_2023():
    d = pd.read_csv(DATA / "airfare-ml/Cleaned_dataset.csv")
    # Scraped_dataset.csv is row-aligned with the cleaned file (checked: flight code and price match 100%)
    sc = pd.read_csv(DATA / "airfare-ml/Scraped_dataset.csv")
    d["dep_hour"] = _hour(sc["Departure Time"])
    d["arr_hour"] = _hour(sc["Arrival Time"])
    d = d.drop_duplicates(subset=[c for c in d.columns if c not in ("dep_hour", "arr_hour")])
    return pd.DataFrame({
        "dep_hour": d.dep_hour, "arr_hour": d.arr_hour,
        "period": 2023,
        "airline": d.Airline.replace(AIRLINE),
        "flight_code": d.Flight_code,
        "source": d.Source, "destination": d.Destination,
        "class": d.Class, "stops": d.Total_stops.map(STOPS),
        "dep_bucket": d.Departure.replace({"Before 6 AM": "Before 6 AM"}),
        "arr_bucket": d.Arrival,
        "duration_h": d.Duration_in_hours, "days_left": d.Days_left,
        "journey_date": pd.to_datetime(d.Date_of_journey),
        "price": d.Fare,
    })


def main():
    df = pd.concat([load_2022(), load_2023()], ignore_index=True)
    df["dow"] = df.journey_date.dt.dayofweek
    df["month"] = df.journey_date.dt.month
    df["route"] = df.source + "-" + df.destination
    df = df.dropna(subset=["price", "stops", "dep_bucket", "arr_bucket"])
    df.to_parquet(OUT)
    print(df.groupby("period").agg(rows=("price", "size"), mean=("price", "mean"),
                                   dmin=("journey_date", "min"), dmax=("journey_date", "max")))
    print(df.groupby(["period", "class"]).price.agg(["size", "mean", "median"]).round(0))
    print(df.groupby(["period", "airline"]).price.size().unstack(0))


if __name__ == "__main__":
    main()
