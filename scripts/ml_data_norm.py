import pandas as pd
import numpy as np
from mne.externals.pymatreader import read_mat
from mne.filter import filter_data
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, lfilter
from sklearn.utils import shuffle
from tqdm import tqdm

data_path = 'E:/YandexDisk/EEG/raw/'

data_file = '1st_Day.mat'
dataframe_path = 'E:/YandexDisk/EEG/Dataframes/'

data = {}
data_indexes = {}
curr_mat_data = read_mat(data_path + data_file)
file_data_flag = ''
if 'subs_ica' in curr_mat_data:
    file_data_flag = 'subs_ica'
else:
    file_data_flag = 'res'

electrodes_names = curr_mat_data[file_data_flag][0]['right_real']['label']
sample_frequency = curr_mat_data[file_data_flag][0]['right_real']['fsample']
for subject_id in range(0, len(curr_mat_data[file_data_flag])):
    for key in curr_mat_data[file_data_flag][subject_id]:
        if key not in data:
            data[key] = []
            data_indexes[key] = []
        for trial_id in range(0, len(curr_mat_data[file_data_flag][subject_id][key]['trial'])):
            data[key].append(curr_mat_data[file_data_flag][subject_id][key]['trial'][trial_id] * 10 ** 6)
            data_indexes[key].append(f"S{subject_id}_T{trial_id}_{key}")
del curr_mat_data

movements = list(data.keys())
freq_bands = [('Alpha', 8, 12)]


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


features_dict = {}
features_dict_norm = {}
features_dict_add = {}
features_dict_names_alpha = {'features': []}

for movement in data:
    print(movement)

    for trial_id in tqdm(range(0, len(data[movement]))):
        curr_index = data_indexes[movement][trial_id]
        if 'index' not in features_dict:
            features_dict['index'] = [curr_index]
        else:
            features_dict['index'].append(curr_index)
        curr_data = data[movement][trial_id]

        for freq_band in freq_bands:
            band_name = freq_band[0]
            band_low = freq_band[1]
            band_high = freq_band[2]

            for electrode_id in range(0, len(electrodes_names)):
                electrode_name = electrodes_names[electrode_id]

                psds, freqs = psd_array_multitaper(x=curr_data[electrode_id, 5500:9500], sfreq=sample_frequency,
                                                   fmin=band_low, fmax=band_high,
                                                   adaptive=True, normalization='full', verbose=0)
                curr_psd = np.trapz(psds, freqs)

                psd_electrode_key = f"{electrode_name}_{band_name}_PSD"
                psd_norm_electrode_key = f"{electrode_name}_{band_name}_PSD_ext"

                if trial_id == 0 and movement == movements[0]:
                    features_dict_names_alpha['features'].append(psd_norm_electrode_key)
                if psd_electrode_key not in features_dict:
                    features_dict[psd_electrode_key] = [curr_psd]
                    features_dict_norm[psd_norm_electrode_key] = [curr_psd]
                else:
                    features_dict[psd_electrode_key].append(curr_psd)
                    features_dict_norm[psd_norm_electrode_key].append(curr_psd)

        if 'class' not in features_dict_add:
            features_dict_add['class'] = [movement]
        else:
            features_dict_add['class'].append(movement)
        if 'class_simp' not in features_dict_add:
            if '1st' in data_file and 'im' in movement:
                features_dict_add['class_simp'] = [movement[:-1]]
            else:
                features_dict_add['class_simp'] = [movement]
        else:
            if '1st' in data_file and 'im' in movement:
                features_dict_add['class_simp'].append(movement[:-1])
            else:
                features_dict_add['class_simp'].append(movement)
        if 'subject' not in features_dict_add:
            features_dict_add['subject'] = [curr_index.split('_')[0]]
        else:
            features_dict_add['subject'].append(curr_index.split('_')[0])

features_df_norm = pd.DataFrame.from_dict(features_dict_norm)
norm_df = ((features_df_norm - features_df_norm.min().min()) / (
            features_df_norm.max().max() - features_df_norm.min().min()))

add_df = pd.DataFrame.from_dict(features_dict_add)

features_df_raw = pd.DataFrame.from_dict(features_dict)
features_all_df = pd.concat([features_df_raw, norm_df], axis=1)

features_df = pd.concat([features_all_df, add_df], axis=1)
features_df = shuffle(features_df)
features_df.reset_index(inplace=True, drop=True)
features_df.to_excel(f"{dataframe_path}/data_alpha_ext.xlsx", header=True, index=False)

features_names_alpha_df = pd.DataFrame.from_dict(features_dict_names_alpha)
features_names_alpha_df.to_excel(f"{dataframe_path}/features_alpha_ext.xlsx", header=True, index=False)
