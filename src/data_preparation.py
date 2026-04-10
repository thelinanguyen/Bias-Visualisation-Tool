# This file contains the data loading, cleaning, encoding, and splitting logic for the Adult Census Dataset.

from pathlib import Path
import time

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


_CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"
_CACHE_PATH = _CACHE_DIR / "adult_cleaned.csv"


def _fetch_adult_raw_data():
  """Fetch Adult dataset with retries and fallbacks.

  Returns a dataframe containing features and an `income` target column.
  """
  last_error = None

  # Primary source: ucimlrepo with simple retry for transient 5xx issues.
  for attempt in range(3):
    try:
      adult = fetch_ucirepo(id=2)
      X = adult.data.features.copy()
      y = adult.data.targets.copy()
      return pd.concat([X, y], axis=1)
    except Exception as err:
      last_error = err
      if attempt < 2:
        time.sleep(1.5 * (attempt + 1))

  # Fallback source: OpenML mirror.
  try:
    openml = fetch_openml(name="adult", version=2, as_frame=True)
    X = openml.data.copy()
    y = openml.target.copy()
    return pd.concat([X, y.rename("income")], axis=1)
  except Exception as err:
    last_error = err

  # Final fallback: most recent successful cached copy.
  if _CACHE_PATH.exists():
    return pd.read_csv(_CACHE_PATH)

  raise ConnectionError(
    "Unable to download Adult dataset from UCI and OpenML, and no local cache exists. "
    f"Last error: {last_error}"
  )

def get_clean_census_data():
  
  """Fetches, cleans, encodes, and splits the Adult Census Dataset. """
    
  # Fetch the data
  df_combined = _fetch_adult_raw_data()

  # Keep feature split explicit for the next processing steps
  X = df_combined.drop(columns=['income'])

  # Handle the "?" symbol (the missing values)
  X = X.replace('?', np.nan)
  df_cleaned = df_combined.dropna()

  # Remove Redundant/Irrelevant Columns
  # Dropping "education" (becuase we have education-num) and "fnlwgt" 
  df_cleaned = df_cleaned.drop(columns=['education', 'fnlwgt'])
  df_cleaned['income'] = df_cleaned['income'].str.strip('.')
  
  # Strip whitespace from all string columns to ensure consistency
  for col in df_cleaned.select_dtypes(include=['object']).columns:
      df_cleaned[col] = df_cleaned[col].str.strip()

  # Save a cleaned cache for offline or outage scenarios.
  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  df_cleaned.to_csv(_CACHE_PATH, index=False)

  # Split back into X and y
  X_final = df_cleaned.drop(columns=['income'])
  y_final = df_cleaned['income']


  # Apply one-hot encoding to categorical features (columns)
  X_encoded = pd.get_dummies(X_final)

  # Ensure the target 'y' is also 0 and 1.Current income is '>50K' or '<=50K'
  y_encoded = y_final.apply(lambda x: 1 if '>50K' in x else 0)

  # Define the features we want to monitor for bias
  protected_attributes = ['sex_Female', 'sex_Male', 'race_White', 'race_Black', 'race_Other']

  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(
      X_encoded, 
      y_encoded, 
      test_size=0.2, 
      random_state=42, 
      stratify=y_encoded  
  )
  
  return X_train, X_test, y_train, y_test, protected_attributes, df_cleaned

if __name__ == "__main__":
  X_train, X_test, y_train, y_test, protected_attributes, _ = get_clean_census_data()
  print(f"Data successfully loaded and split! Training rows: {len(X_train)}")
  print(f"Protected attributes: {protected_attributes}")

