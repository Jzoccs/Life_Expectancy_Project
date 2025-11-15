import requests
import pandas as pd
from pathlib import Path


def main():
    # 1. Choose the indicator and build the URL
    # Life expectancy at birth, total (years)
    indicator = "SP.DYN.LE00.IN"

    # World Bank API endpoint:
    # https://api.worldbank.org/v2/country/all/indicator/<indicator>?format=json&per_page=20000
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator}?format=json&per_page=20000"
    )

    # 2. Call the API (like opening a web page, but in code)
    response = requests.get(url)

    # This will raise an error if something went wrong (e.g., no internet)
    response.raise_for_status()

    # 3. Convert the response (JSON text) into Python objects (list/dicts)
    data = response.json()

    # The World Bank returns a list: [metadata, actual_data]
    # data[0] = metadata (how many pages, etc.)
    # data[1] = list of records we actually care about
    records = data[1]

    # 4. Turn the JSON records into a pandas DataFrame
    df = pd.json_normalize(records)

    # 5. Keep and rename a few useful columns
    df = df[["countryiso3code", "country.value", "date", "value"]].rename(
        columns={
            "countryiso3code": "iso3",
            "country.value": "country_name",
            "date": "year",
            "value": "life_expectancy",
        }
    )

    # Make year an integer instead of text
    df["year"] = df["year"].astype(int)

    # Show the first 5 rows so you can see what it looks like
    print(df.head())

    # 6. Save to data/raw/worldbank_life_expectancy_simple.csv
    output_path = Path("data") / "raw" / "worldbank_life_expectancy_simple.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
