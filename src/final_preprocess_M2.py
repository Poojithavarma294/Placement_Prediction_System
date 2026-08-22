# Placement Prediction Dataset Preprocessing
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------
# 1. Read Original Dataset
# ------------------------------------------------------------

input_file = r"C:/Users/Dell/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"

output_file = r"C:/Users/Dell/PycharmProjects/placement_prediction/dataset/final_preprocess_M2.csv"

df_original = pd.read_csv(input_file)

print("=" * 70)
print("PLACEMENT PREDICTION DATASET PREPROCESSING")
print("=" * 70)

print("\nOriginal Dataset Shape:", df_original.shape)


# ------------------------------------------------------------
# 2. Create Copy
# ------------------------------------------------------------

processed_df = df_original.copy()


# ------------------------------------------------------------
# 3. Handle Missing Values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

print("\nMissing values before preprocessing:")
print(processed_df.isnull().sum())

# ------------------------------------------------------------
# Numerical Columns
# Fill missing values using Median
# ------------------------------------------------------------

numeric_cols = processed_df.select_dtypes(
    include=['number']
).columns

for col in numeric_cols:

    if processed_df[col].isnull().any():

        median_value = processed_df[col].median()

        processed_df[col] = processed_df[col].fillna(
            median_value
        )
# ------------------------------------------------------------
# Categorical Columns
# Fill missing values using Mode
# ------------------------------------------------------------

categorical_cols = processed_df.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_cols:

    if processed_df[col].isnull().any():

        mode_values = processed_df[col].mode()

        if not mode_values.empty:

            processed_df[col] = processed_df[col].fillna(
                mode_values.iloc[0]
            )


print("\nMissing values after preprocessing:")
print(processed_df.isnull().sum())

print("\n" + "=" * 70)
print("CATEGORICAL ENCODING")
print("=" * 70)

label_encoders = {}

categorical_cols = processed_df.select_dtypes(
    include=['object', 'string']
).columns

for col in categorical_cols:

    le = LabelEncoder()

    processed_df[col] = le.fit_transform(
        processed_df[col].astype(str)
    )

    label_encoders[col] = le

    print(f"Encoded column: {col}")


# ------------------------------------------------------------
# 5. Feature Scaling
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

numeric_cols = processed_df.select_dtypes(
    include=['number']
).columns

scaler = StandardScaler()

processed_df[numeric_cols] = scaler.fit_transform(
    processed_df[numeric_cols]
)

print("Numerical features standardized successfully.")


# ------------------------------------------------------------
# 6. Final Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET INFORMATION")
print("=" * 70)

print("\nOriginal Dataset Shape :", df_original.shape)
print("Processed Dataset Shape:", processed_df.shape)

print("\nRemaining Missing Values:")
print(processed_df.isnull().sum().sum())


# ------------------------------------------------------------
# 7. Save Processed Dataset
# ------------------------------------------------------------

processed_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nOriginal Dataset Shape :", df_original.shape)
print("Processed Dataset Shape:", processed_df.shape)
print("Saved File :", output_file)