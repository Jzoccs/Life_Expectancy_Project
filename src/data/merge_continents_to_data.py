import pandas as pd
from pathlib import Path

# ---------------------------------------------
# Paths
# ---------------------------------------------
project_root = Path(__file__).resolve().parents[2]

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"

cleaned_path = processed_dir / "cleaned_life_expectancy_gdp_co2.csv"
continents_path = raw_dir / "continents.csv"

output_path = processed_dir / "merged_dataset_with_continents.csv"

# ---------------------------------------------
# Load datasets
# ---------------------------------------------
df = pd.read_csv(cleaned_path)
df_cont = pd.read_csv(continents_path)

# ---------------------------------------------
# Merge Continents (ONLY on iso3)
# ---------------------------------------------
df = df.merge(df_cont, on="iso3", how="left")

# ---------------------------------------------
# Save output
# ---------------------------------------------
processed_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Saved merged dataset to: {output_path}")
