import pandas as pd
import numpy as np

df = pd.read_csv(r"data/clusters/patient_features_CAD_train.csv")   # <-- change to your file name
# Keep first 8 (index 0 to 7) and last 2 columns
df_subset = df.iloc[:, np.r_[0:48, -2:0]]
print(df_subset.head())
df_subset.to_csv(r"data/clusters/patient_features_CAD_train_subset.csv", index=False)


