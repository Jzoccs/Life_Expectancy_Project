import requests
import pandas as pd
from pathlib import Path


def main():
    # World Bank indicator for GDP per capita (constant 2015 US$)
    indicator = "NY.GDP.PCAP.KD"

    # Build the API URL
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{indicator}?format=json&per_page=20000"
    )

    # Call the API
    response = requests.get(url)
    response.raise_for_status()

    # Parse JSON
    data = response.json()
    records = data[1]  # actual records

    # Convert to DataFrame
    df = pd.json_normalize(records)

    # Keep and rename the useful columns
    df = df[["countryiso3code", "country.value", "date", "value"]].rename(
        columns={
            "countryiso3code": "iso3",
            "country.value": "country_name",
            "date": "year",
            "value": "gdp_per_capita",
        }
    )

    # Convert year to integer
    df["year"] = df["year"].astype(int)

    # Quick peek
    print("GDP per capita sample:")
    print(df.head(), "\n")

    # Save to data/raw
    output_path = Path("data") / "raw" / "worldbank_gdp_per_capita_simple.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
