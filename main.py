import pandas as pd
import cleaning as cl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from clustering import elbow_method, kmeans
import seaborn as sns

def create_visual(data, pca, save_plots=True):

  plt.style.use('seaborn-v0_8')

  # 3D PCA Cluster Plot
  fig = plt.figure(figsize=(12, 8))
  ax = fig.add_subplot(111, projection='3d')

  colors = plt.cm.Set1(np.linspace(0, 1, data['Cluster'].nunique()))

  for i, cluster in enumerate(sorted(data['Cluster'].unique())):
    cluster_data = pca[data['Cluster'] == cluster]
    ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
               c=[colors[i]], label=f'Cluster {cluster}', alpha=0.7, s=50)

  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  ax.set_zlabel('PC3')
  ax.set_title('3D PCA Cluster Visualization')
  ax.legend()

  if save_plots:
    plt.savefig('./plots/cluster/pca_3d_clusters.png', dpi=300, bbox_inches='tight')
  else:
    plt.show()

  # 2D Cluster plots
  fig, axes = plt.subplots(2, 2, figsize=(15, 12))

  plot_pairs = [
    ('Income', 'Tot_Mnt', 'Income vs Total Spending'),
    ('Age', 'Tot_Mnt', 'Age vs Total Spending'),
    ('Income', 'Tot_Accepted', 'Income vs Campaigns Accepted'),
    ('Dt_Customer', 'Tot_Mnt', 'Customer Tenure vs Spending')
  ]

  for idx, (x, y, title) in enumerate(plot_pairs):
    ax = axes[idx // 2, idx % 2]

    for cluster in sorted(data['Cluster'].unique()):
      cluster_data = data[data['Cluster'] == cluster]
      ax.scatter(cluster_data[x], cluster_data[y],
                 label=f'Cluster {cluster}', alpha=0.6, s=30)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

  plt.tight_layout()
  if save_plots:
    plt.savefig('./plots/cluster/cluster_analysis_plots.png', dpi=300, bbox_inches='tight')
  else:
    plt.show()


  plt.figure()
  pl = sns.swarmplot(x=data['Cluster'], y=data["MntWines"], alpha=0.5)
  pl = sns.boxenplot(x=data['Cluster'], y=data['MntWines'])
  if save_plots:
    plt.savefig('./plots/cluster/cluster_wines.png', dpi=500)
  else:
   plt.show()

  plt.figure()
  pl = sns.swarmplot(x=data['Cluster'], y=data["MntMeatProducts"], alpha=0.5)
  pl = sns.boxenplot(x=data['Cluster'], y=data['MntMeatProducts'])
  if save_plots:
    plt.savefig('./plots/cluster/cluster_meat.png', dpi=500)
  else:
    plt.show()

  To_Plot = ["Dt_Customer", "Age", "Parent", "Education_High", "Education_Low", "Marital_Status_Partner", "Tot_Purchase", "Children"]
  for i in To_Plot:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Tot_Mnt"], hue=data["Cluster"], kind="kde", palette="viridis", alpha=0.5)
    if save_plots:
      plt.savefig(f"./plots/cluster/cluster_totmnt_{i}.png", dpi=500)
    else:
      plt.show()

def main():
  try:
    df = pd.read_csv("./data/marketing_campaign.csv", sep="\t")
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
  except FileNotFoundError:
    print("Error: marketing_campaign.csv not found in ./data/ directory")
    return

  # handling missing values
  initial_rows = df.shape[0]
  df.dropna(inplace=True)
  print(f"Rows after dropping nulls: {df.shape[0]} (removed {initial_rows - df.shape[0]})")

  # drop const columns
  constant_cols = ["Z_CostContact", "Z_Revenue"]
  df.drop(columns=[col for col in constant_cols if col in df.columns], inplace=True)

  # feature changes
  df['Age'] = 2025 - df['Year_Birth']
  df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
  df['Dt_Customer'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days
  df['Children'] = df['Teenhome'] + df['Kidhome']
  df['Parent'] = np.where(df['Children'] > 0, 1, 0)

  # clean meaningless categorical data
  df = df[(df['Marital_Status'] != 'YOLO') & (df['Marital_Status'] != 'Absurd')]

  # reduce categories
  df["Marital_Status"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Widow": "Alone", "Divorced": "Alone", "Single": "Alone"
  })

  df['Education'] = df['Education'].replace({
    'Graduation': 'Low', 'PhD': 'High', 'Master': 'High', '2n Cycle': 'Low'
  })

  # create aggregate features
  mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
  purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
  campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

  df['Tot_Mnt'] = df[mnt_cols].sum(axis=1)
  df['Tot_Purchase'] = df[purchase_cols].sum(axis=1)
  df['Tot_Accepted'] = df[campaign_cols].sum(axis=1)

  # outlier removal
  df = df[df['Age'] < 90]
  df = df[df['Income'] < 600000]
  for col in mnt_cols:
    if col in df.columns:
      df = cl.iqr_filter(df, col, rate=4.0, verbose=True)

  # one-hot encoding
  df_encoded = cl.one_hot_encoding(df, ['Education', 'Marital_Status'])

  # drop unnecessary columns
  drop_cols = ['Year_Birth', 'ID'] + campaign_cols
  df_encoded = df_encoded.drop([col for col in drop_cols if col in df_encoded.columns], axis=1)

  # features for clustering
  feature_cols = [
    'Age', 'Income', 'Tot_Mnt', 'Tot_Purchase', 'Dt_Customer',
    'Education_High', 'Parent', 'Marital_Status_Partner', 'Tot_Accepted'
  ]
  existing_features = [col for col in feature_cols if col in df_encoded.columns]
  print(f"\nUsing features for clustering: {existing_features}")

  # scale the encoded dataframe
  X = df_encoded[existing_features].values
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  clusters, centroids, wcss = kmeans(X_scaled, 4, kmeans_pp=True)
  df_encoded['Cluster'] = clusters

  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X_scaled)

  df_encoded['PC1'] = X_pca[:, 0]
  df_encoded['PC2'] = X_pca[:, 1]
  df_encoded['PC3'] = X_pca[:, 2]

  create_visual(df_encoded, X_pca)

if __name__ == '__main__':
    main()