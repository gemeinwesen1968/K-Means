import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

def print_row_amount(data):
  print('amount of data points:', data.shape[0], '\n')

def iqr_filter(data: DataFrame, col: str, rate: float = 1.5, inplace: bool = False, verbose: bool = False):
  if rate <= 0:
    raise ValueError("Rate value must be a positive float.")
  q1 = data[col].quantile(0.25)
  q3 = data[col].quantile(0.75)
  iqr = q3 - q1
  lower_bound = q1 - rate * iqr
  upper_bound = q3 + rate * iqr
  if lower_bound < 0:
    lower_bound = 0
  mask = (data[col] >= lower_bound) & (data[col] <= upper_bound)
  removed = data.shape[0] - mask.sum()
  if verbose:
    print(f"{col}: removed {removed} outliers")
    print(f"lower and upper bound: {lower_bound:.2f} - {upper_bound:.2f} (rate {rate})")
    print('Remaining:', mask.sum(), '\n')
  if inplace:
    data.drop(index=data[~mask].index, inplace=True)
    return None
  else:
    return data[mask].copy()

def one_hot_encoding(data: DataFrame, columns: list[str]):
  df_encoded = pd.get_dummies(data, columns=columns)
  bool_cols = df_encoded.select_dtypes(include='bool').columns
  df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
  return df_encoded

def scale_filter(data: DataFrame, columns: list[str]):
  scaler = MinMaxScaler()
  data[columns] = scaler.fit_transform(data[columns])