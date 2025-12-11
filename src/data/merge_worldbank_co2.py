import os
import pandas as pd

'''
This merges three csv's into one processed csv that we will then perform cleaning on
'''
def main():
    #1. Read the two World Bank CSVs
    life_df = pd.read_csv("data/raw/worldbank_life_expectancy_simple.csv")
    gdp_df = pd.read_csv("data/raw/worldbank_gdp_per_capita_simple.csv")

    #Merge life + GDP on country + year
    wb_df = life_df.merge(
        gdp_df,
        on=["iso3", "country_name", "year"],
        how="inner",
    )

    #2. Read the CO2 data from OWID (downloaded to data/external/owid-co2-data.csv)
    co2_df = pd.read_csv("data/external/owid-co2-data.csv")

    #Keep only the columns we need and rename iso_code -> iso3
    co2_df = co2_df[["iso_code", "year", "co2_per_capita"]].rename(
        columns={"iso_code": "iso3"}
    )

    #Make sure year is integer
    co2_df["year"] = co2_df["year"].astype(int)

    #3. Merge World Bank data with CO2 data on iso3 and year
    full_df = wb_df.merge(
        co2_df,
        on=["iso3", "year"],
        how="inner",
    )

    #4. Save the final merged dataset
    os.makedirs("data/processed", exist_ok=True)
    full_df.to_csv("data/processed/life_expectancy_gdp_co2.csv", index=False)

    print("Saved merged dataset to data/processed/life_expectancy_gdp_co2.csv")


if __name__ == "__main__":
    main()
