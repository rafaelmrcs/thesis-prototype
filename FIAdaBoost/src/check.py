import pandas as pd

df = pd.read_csv("data/processed/baseline_spatial_clean_2024.csv")

print(df["GHI_mean_2024"].describe())
print("Unique values:", df["GHI_mean_2024"].nunique())
