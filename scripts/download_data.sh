#!/usr/bin/env bash
# Download the Kaggle datasets used by the project into ./data.
# Needs the Kaggle CLI and your own API token in ~/.kaggle/kaggle.json (or ~/.kaggle/access_token).
# Get a token at https://www.kaggle.com/settings -> API. Never commit it.
set -euo pipefail
cd "$(dirname "$0")/.."

dl() {  # dl <kaggle-slug> <folder>
  mkdir -p "data/$2"
  (cd "data/$2" && kaggle datasets download "$1" --unzip -q)
  echo "ok  $2"
}

# used by the model
dl yashdharme36/airfare-ml-predicting-flight-fares airfare-ml
dl rohitgrewal/airlines-flights-data airlines-flights

# downloaded for exploration, not used by the model yet
dl anshuman0427/flight-price-dataset flight-price-2019
dl andrewgeorgeissac/flight-price-data-of-indian-cities-makemytrip makemytrip
dl kabil007/indian-domestic-airline-dataset indian-domestic-schedules
