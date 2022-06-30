import pandas as pd
import pickle

experiments = ['1st_day', '2nd_day_sham', '2nd_day_tms']

for experiment in experiments:
    data_path = f'E:/YandexDisk/EEG/experiments/{experiment}/'
    data_df = pd.read_excel(f'{data_path}data.xlsx')
    with open(f'{data_path}data.pickle', 'wb') as handle:
        pickle.dump(data_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
