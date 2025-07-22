import pandas as pd
import cleaning as cl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from clustering import elbow_method, kmeans

# checks given column name lists
# returns names present
def check_columns(data, cols_to_check):
  return [col for col in cols_to_check if col in data.columns]

df = pd.read_csv('./data/marketing_campaign.csv', sep='\t')

# drop null rows
df.dropna(inplace=True)

# Z_CostContact and Z_Revenue are same for all rows
df.drop(columns=['Z_CostContact',	'Z_Revenue'], inplace=True)

df['Age'] = 2025 - df['Year_Birth']

# turn Dt_Customer into 'days as a customer'
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Dt_Customer'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days

# limit age and income to drop outliers
df = df[df['Age'] < 90]
df = df[df['Income'] < 600000]

df = df[(df['Marital_Status'] != 'YOLO') & (df['Marital_Status'] != 'Absurd')]

# amount of children
df['Children'] = df['Teenhome'] + df['Kidhome']

# reduce "Marital_Status" into binary (Partner or Alone)
df["Marital_Status"]=(df["Marital_Status"]
                     .replace({"Married":"Partner",
                               "Together":"Partner",
                               "Widow":"Alone",
                               "Divorced":"Alone",
                               "Single":"Alone",}))

#reduce "Education" into binary (Low or High)
df['Education'] = df['Education'].replace({'Graduation':'Low',
                                           'PhD':'High',
                                           'Master':'High',
                                           '2n Cycle': 'Low'})

df['Parent'] = np.where(df.Children > 0, 1, 0)

numeric = ['Age', 'Kidhome', 'Teenhome', 'Children', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'Dt_Customer', 'Income']
mnt = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
encoding = ['Education', 'Marital_Status']

purchase = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
accepted = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
df['Tot_Mnt'] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]
df['Tot_Purchase'] = df[check_columns(df, mnt)].sum(axis=1)
df['Tot_Accepted'] = df[check_columns(df, accepted)].sum(axis=1)
numeric.append('Tot_Mnt')
numeric.append('Tot_Purchase')
numeric.append('Tot_Accepted')

# filter mnt values using quartiles
for column in check_columns(df, mnt):
   cl.iqr_filter(df, column, 4.0, verbose=True, inplace=True)

df = cl.one_hot_encoding(df, check_columns(df, encoding))

to_drop = ['Year_Birth', 'ID']
df = df.drop(check_columns(df, to_drop), axis=1)
df = df.drop(check_columns(df, accepted), axis=1)

scaled_df = df.copy()
numeric.extend(mnt)

# scale non-binary numeric columns
true_numeric = [col for col in numeric if df[col].nunique() > 2]
cl.scale_filter(scaled_df, check_columns(scaled_df, true_numeric))

categorical_binary = [col for col in scaled_df.columns if scaled_df[col].nunique() == 2]

print("#### SCALED DATA INFO ####")
print(scaled_df.info())

print("#### DATA MEANS ####")
print(df.mean())
print("#### DATA MEDIANS ####")
print(df.median())

#fig, axes = plt.subplots(1, 5, figsize=(18, 5))

#sns.histplot(df['Age'], kde=True, bins=30, ax=axes[0])
#axes[0].set_title('Age Dağılımı')
#axes[0].set_xlabel('Age')

#sns.histplot(df['Income'], kde=True, bins=30, ax=axes[1])
#axes[1].set_title('Income Dağılımı')
#axes[1].set_xlabel('Income')
#
#sns.histplot(df['Recency'], kde=True, bins=30, ax=axes[2])
#axes[2].set_title('Recency Dağılımı')
#axes[2].set_xlabel('Recency')
#
#sns.histplot(df['Dt_Customer'], kde=True, bins=30, ax=axes[3])
#axes[3].set_title('Dt_Customer Dağılımı')
#axes[3].set_xlabel('Days Since Customer')
#
#sns.histplot(df['Tot_Mnt'], kde=True, bins=30, ax=axes[4])
#axes[4].set_title('Tot_Mnt Dağılımı')
#axes[4].set_xlabel('Tot_Mnt')
#
#plt.tight_layout()
#plt.savefig("./plots/distribution.png", dpi=500)

#mnt_sums = df[mnt].sum()
#plt.figure(figsize=(8, 8))
#plt.pie(mnt_sums, labels=mnt, autopct='%1.1f%%', startangle=140)
#plt.title('Ürün Harcamalarının Dağılımı (%)')
#plt.axis('equal')
#plt.savefig("./plots/mnt_distribution_pie.png", dpi=500)
#

def auto_scatter_plot(data, x, y, ytitle, hue, nrows, ncols, flatten=True):
  fig, xes = plt.subplots(nrows, ncols, figsize=(18, 10))
  if flatten:
    xes = xes.flatten()
  for i, col in enumerate(y):
    sns.scatterplot(
      data=df, x=x, ax=xes[i], y=col,
      hue=data[hue], alpha=0.6
    )
    xes[i].set_title(f'{x} vs {col} by {hue}')
    xes[i].legend(title=hue)
  plt.tight_layout()
  plt.savefig(f'./plots/{x}_{ytitle}_scatter_by_{hue}.png', dpi=500)

hues = ['Parent', 'Education_High', 'Marital_Status_Partner']
xlist = ['Age', 'Income', 'Dt_Customer', 'Recency']
ylist = [[mnt, 'mnt'], [purchase, 'purchase']]

#for y in ylist:
#  for x in xlist:
#    for hue in hues:
#      ncol = 2 if y[1] == 'purchase' else 3
#      auto_scatter_plot(df, x, y[0], y[1], hue, 2, ncol)

#To_Plot = [ "Income", "Recency", "Age", "Tot_Mnt"]
#for hue in hues:
#  To_Plot_hue = list(set(To_Plot + [hue]))
#  plt.figure()
#  sns.pairplot(df[To_Plot_hue], hue=hue)
#  plt.savefig(f'./plots/pairplot_{hue}.png', dpi=500)

#corrmat= df.corr()
#plt.figure(figsize=(20,20))
#sns.heatmap(corrmat,annot=True, cmap='viridis', center=0)
#plt.savefig('./plots/corrmat.png', dpi=500)

#corrmat= scaled_df.corr()
#plt.figure(figsize=(20,20))
#sns.heatmap(corrmat,annot=True, cmap='viridis', center=0)
#plt.savefig('./plots/scaled_corrmat.png', dpi=500)

X = scaled_df[["Age", "Income", "Tot_Mnt", "Tot_Purchase",
               "Dt_Customer", "Education_High", "Parent",
               "Marital_Status_Partner", "Tot_Accepted"]].values

elbow_method(X)
clusters, centroids, _ = kmeans(X, 4, kmeans_pp=True)
scaled_df["Cluster"] = clusters

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
scaled_df["PC1"] = X_pca[:, 0]
scaled_df["PC2"] = X_pca[:, 1]
scaled_df["PC3"] = X_pca[:, 2]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    scaled_df["PC1"], scaled_df["PC2"], scaled_df["PC3"],
    c=scaled_df["Cluster"], cmap='Set1', alpha=0.6
)

ax.set_title("3D PCA Cluster Plot")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
plt.savefig("./plots/pca_3d_clusters_pp.png", dpi=500)

plt.figure(figsize=(10, 6))
pl = sns.scatterplot(data=scaled_df, x="Income", y="Tot_Mnt", hue="Cluster", palette="Accent")
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.savefig("./plots/clusters_income_spending.png", dpi=500)