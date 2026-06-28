import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("german_credit_data.csv")

print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values Before Filling:")
print(df.isnull().sum())

# ==========================
# Remove unwanted column
# ==========================
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ==========================
# Fill Missing Values
# ==========================
for col in df.columns:
    if is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Filling:")
print(df.isnull().sum())

# ==========================
# Encode Categorical Columns
# ==========================
encoder = LabelEncoder()

for col in df.columns:
    if not is_numeric_dtype(df[col]):
        df[col] = encoder.fit_transform(df[col])

print("\nDataset After Encoding:")
print(df.head())

# ==========================
# Save Clean Dataset
# ==========================
df.to_csv("cleaned_german_credit_data.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ==========================
# Basic Statistics
# ==========================
print("\nDataset Statistics:")
print(df.describe())

# ==========================
# Graph 1 - Age Distribution
# ==========================
plt.figure(figsize=(7,5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()

# ==========================
# Graph 2 - Credit Amount Distribution
# ==========================
plt.figure(figsize=(7,5))
plt.hist(df["Credit amount"], bins=20)
plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount")
plt.ylabel("Number of Customers")
plt.show()

# ==========================
# Graph 3 - Housing Type
# ==========================
plt.figure(figsize=(6,4))
df["Housing"].value_counts().plot(kind="bar")
plt.title("Housing Type")
plt.xlabel("Housing")
plt.ylabel("Count")
plt.show()

print("\nProject executed successfully!")

print("\nNOTE:")
print("This dataset does not contain a target column like 'Risk' or 'Loan_Status'.")
print("Therefore, a machine learning credit scoring model cannot be trained using this dataset.")
print("To complete Task 1, you need a dataset containing a target column such as 'Risk'.")