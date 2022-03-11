import numpy as np
import pandas as pd
from pathlib import Path


data_path = 'E:/YandexDisk/EEG/Dataframes/'
path = 'E:/YandexDisk/EEG/'
data_file = 'dataframe_1st_Day.xlsx'

classes = ['right_im2', 'left_im2', 'background']
suffix = f"1st_Day_{'_'.join(classes)}"
Path(f"{path}Files/{suffix}/").mkdir(parents=True, exist_ok=True)

df = pd.read_excel(data_path + data_file)
df_classes = df[df['class'].str[:].isin(classes)]
df_classes = df_classes.rename(columns={'trial': 'index'})
subjects_names = [item.split('_')[0] for item in list(df_classes['index'])]
df_classes['subject'] = subjects_names

test_subjects = ['S5', 'S6', 'S7', 'S8']
mask = df_classes['index'].str[:2].isin(test_subjects)
df_test = df_classes[mask]
df_train_val = df_classes[~mask]

features_names = list(df_train_val.columns.values)[1:-2]
features_df = pd.DataFrame.from_dict({'features': features_names})

classes_df = pd.DataFrame.from_dict({'class': classes})

df_train_val.to_excel(f"{path}Files/{suffix}/train_val_df.xlsx", header=True, index=False)
df_test.to_excel(f"{path}Files/{suffix}/test_df.xlsx", header=True, index=False)
features_df.to_excel(f"{path}Files/{suffix}/features_df.xlsx", header=True, index=False)
classes_df.to_excel(f"{path}Files/{suffix}/classes_df.xlsx", header=True, index=False)
