import pandas as pd
import cleaning as cl

df = pd.read_csv('./data/marketing_campaign.csv', sep='\t')

df.dropna(inplace=True)
df.drop(columns=['Z_CostContact',	'Z_Revenue'], inplace=True)

df['Age'] = 2025 - df['Year_Birth']

numeric_1 = ['Age', 'Kidhome', 'Teenhome', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
numeric_2 = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
encoding_columns = ['Education', 'Marital_Status']

for column in numeric_1:
   cl.iqr_filter(df, column, 1.5, verbose=True, inplace=True)

for column in numeric_2:
   cl.iqr_filter(df, column, 3.0, verbose=True, inplace=True)

df = cl.one_hot_encoding(df, encoding_columns)

df.drop(columns=['Marital_Status_YOLO'], inplace=True)

numeric_1.extend(numeric_2)
cl.scale_filter(df, numeric_1)

print(df.info())