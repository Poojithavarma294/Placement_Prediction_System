import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("C:/Users/dell/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv")

print("----First 5 Rows -----")
print(df.head())


print("----print 6 columns----")
subset = df.iloc[:, 0:6]
print(subset)

missing_columns = subset.isnull().sum()
print("-----Missing Values Per Column:----------")

print(missing_columns)
print("-" * 40)

duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)

plt.figure(figsize=(10, 6))

sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

