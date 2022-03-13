import pandas as pd


data_path = 'E:/YandexDisk/EEG/Dataframes/'
data_file = 'dataframe_DAY2_SHAM_uV.xlsx'
data_file_background = 'dataframe_DAY2_SHAM_background_uV.xlsx'

df = pd.read_excel(data_path + data_file)
df_background = pd.read_excel(data_path + data_file_background)

df_background['class'] = 'background'
df_background_trials = list(df_background['trial'])
background_trials = []
for trial in df_background_trials:
    background_trials.append(f"{trial}_background")
df_background['trial'] = background_trials

df_final = pd.concat([df, df_background], axis=0)
df_final.to_excel(f"{data_path}/dataframe_2nd_Day_sham_uV.xlsx", header=True, index=False)
