import pandas as pd

# Load questions.csv
df = pd.read_csv("questions.csv")

# Đổi tên branch -> label
df = df.rename(columns={"branch": "label"})[["text", "label"]]
df.to_csv("labeled_data.csv", index=False)
print("labeled_data.csv ready with", len(df), "rows")
