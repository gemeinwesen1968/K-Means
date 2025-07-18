import pandas as pd
import cleaning as cl

def check_columns(data, cols_to_check):
   results = {}
   for col in cols_to_check:
      results[col] = col in data.columns
   return results

df = pd.read_csv('./data/marketing_campaign.csv', sep='\t')

df.dropna(inplace=True)
df.drop(columns=['Z_CostContact',	'Z_Revenue'], inplace=True)

df['Age'] = 2025 - df['Year_Birth']

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
dates = []
for i in df['Dt_Customer']:
   i = i.date()
   dates.append(i)

days = []
d = max(dates)
for i in dates:
   diff = d - i
   days.append(diff)

df['Dt_Customer'] = days
df["Dt_Customer"] = pd.to_numeric(df["Dt_Customer"], errors="coerce")

numeric = ['Age', 'Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'Dt_Customer', 'Income']
mnt = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
encoding = ['Education', 'Marital_Status']

df = df[df['Age'] < 90]
df = df[df['Income'] < 600000]

df = df[(df['Marital_Status'] != 'YOLO') & (df['Marital_Status'] != 'Absurd')]

df["Marital_Status"]=(df["Marital_Status"]
                     .replace({"Married":"Partner",
                               "Together":"Partner",
                               "Widow":"Alone",
                               "Divorced":"Alone",
                               "Single":"Alone",}))

purchase = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df['Mnt_Total'] = df[list(check_columns(df, purchase))].sum(axis=1)

df['Tot_Purchase'] = df[list(check_columns(df, mnt))].sum(axis=1)

filtered_df = df.copy()
for column in list(check_columns(filtered_df, mnt)):
   cl.iqr_filter(filtered_df, column, 4.0, verbose=True, inplace=True)

filtered_df = cl.one_hot_encoding(filtered_df, list(check_columns(filtered_df, encoding)))

numeric.extend(mnt)
numeric.append('Mnt_Total')
numeric.append('Tot_Purchase')
cl.scale_filter(filtered_df, list(check_columns(filtered_df, numeric)))

print(filtered_df.info())