import pandas as pd
import numpy as np
from mne.externals.pymatreader import read_mat
from mne.filter import filter_data
from mne.time_frequency import psd_array_multitaper
from mne_features.univariate import compute_spect_entropy
from scipy.signal import butter, lfilter
from sklearn.utils import shuffle
from tqdm import tqdm
import pickle


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
freq_bands = [('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

brain_regions_dict = {'Frontal': ['FP1', 'FP2', 'F3', 'Fz', 'F4'],
                      'Left-Temporal': ['F7', 'FT9', 'T7', 'TP9', 'P7'],
                      'Left-Central': ['FC5', 'FC1', 'C3', 'CP5', 'CP1', 'Cz'],
                      'Right-Central': ['FC2', 'FC6', 'Cz', 'C4', 'CP2', 'CP6'],
                      'Right-Temporal': ['F8', 'FT10', 'T8', 'TP10', 'P8'],
                      'Occipital': ['P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']}

electrode_region_dict = {}
for key in brain_regions_dict:
    for electrode in brain_regions_dict[key]:
        if electrode not in electrode_region_dict:
            electrode_region_dict[electrode] = [key]
        else:
            electrode_region_dict[electrode].append(key)


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
features_dict_names_all = {'features': []}
features_dict_names_averaged = {'features': []}
features_dict_names_alpha_beta = {'features': []}
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
            filtered_data = butter_bandpass_filter(curr_data, band_low, band_high, sample_frequency)

            spect_ent = compute_spect_entropy(sfreq=sample_frequency, data=filtered_data[:, 5500:9500],
                                              psd_method='multitaper')

            for region in brain_regions_dict:
                curr_region_psd = []
                curr_region_ent = []

                for curr_electrode_id in range(0, len(brain_regions_dict[region])):
                    electrode_name = brain_regions_dict[region][curr_electrode_id]
                    electrode_id = electrodes_names.index(electrode_name)

                    psds, freqs = psd_array_multitaper(x=curr_data[electrode_id, 5500:9500], sfreq=sample_frequency,
                                                       fmin=band_low, fmax=band_high,
                                                       adaptive=True, normalization='full', verbose=0)
                    curr_psd = np.trapz(psds, freqs)
                    curr_region_psd.append(curr_psd)

                    curr_spect_ent = spect_ent[electrode_id]
                    curr_region_ent.append(curr_spect_ent)

                    if region == 'Right-Central' and electrode_name == 'Cz':
                        continue

                    psd_electrode_key = f"{electrode_name}_{band_name}_PSD"
                    se_electrode_key = f"{electrode_name}_{band_name}_SE"

                    if trial_id == 0 and movement == movements[0]:
                        features_dict_names_all['features'].append(psd_electrode_key)
                        features_dict_names_all['features'].append(se_electrode_key)
                        if band_name == 'Alpha':
                            features_dict_names_alpha_beta['features'].append(psd_electrode_key)
                            features_dict_names_alpha_beta['features'].append(se_electrode_key)
                            features_dict_names_alpha['features'].append(psd_electrode_key)
                        elif band_name == 'Beta':
                            features_dict_names_alpha_beta['features'].append(psd_electrode_key)
                            features_dict_names_alpha_beta['features'].append(se_electrode_key)
                    if psd_electrode_key not in features_dict and se_electrode_key not in features_dict:
                        features_dict[psd_electrode_key] = [curr_psd]
                        features_dict[se_electrode_key] = [curr_spect_ent]
                    else:
                        features_dict[psd_electrode_key].append(curr_psd)
                        features_dict[se_electrode_key].append(curr_spect_ent)

                psd_region_key = f"{region}_{band_name}_PSD"
                se_region_key = f"{region}_{band_name}_SE"
                if trial_id == 0 and movement == movements[0]:
                    features_dict_names_averaged['features'].append(psd_region_key)
                    features_dict_names_averaged['features'].append(se_region_key)
                if psd_region_key not in features_dict and se_region_key not in features_dict:
                    features_dict[psd_region_key] = [np.mean(curr_region_psd)]
                    features_dict[se_region_key] = [np.mean(curr_region_ent)]
                else:
                    features_dict[psd_region_key].append(np.mean(curr_region_psd))
                    features_dict[se_region_key].append(np.mean(curr_region_ent))

        for region in brain_regions_dict:
            curr_region_paf = []
            curr_region_iaf = []

            for curr_electrode_id in range(0, len(brain_regions_dict[region])):
                electrode_name = brain_regions_dict[region][curr_electrode_id]
                electrode_id = electrodes_names.index(electrode_name)

                psds, freqs = psd_array_multitaper(x=curr_data[electrode_id, 5500:9500], sfreq=sample_frequency,
                                                   fmin=7, fmax=13,
                                                   adaptive=True, normalization='full', verbose=0)
                curr_paf = freqs[np.argmax(psds)]
                curr_region_paf.append(curr_paf)

                curr_iaf = np.average(freqs, weights=psds)
                curr_region_iaf.append(curr_iaf)

                if region == 'Right-Central' and electrode_name == 'Cz':
                    continue

                paf_electrode_key = f"{electrode_name}_PAF"
                iaf_electrode_key = f"{electrode_name}_IAF"
                if trial_id == 0 and movement == movements[0]:
                    features_dict_names_all['features'].append(paf_electrode_key)
                    features_dict_names_all['features'].append(iaf_electrode_key)
                if paf_electrode_key not in features_dict and iaf_electrode_key not in features_dict:
                    features_dict[paf_electrode_key] = [curr_paf]
                    features_dict[iaf_electrode_key] = [curr_iaf]
                else:
                    features_dict[paf_electrode_key].append(curr_paf)
                    features_dict[iaf_electrode_key].append(curr_iaf)

            paf_region_key = f"{region}_PAF"
            iaf_region_key = f"{region}_IAF"
            if trial_id == 0 and movement == movements[0]:
                features_dict_names_averaged['features'].append(paf_region_key)
                features_dict_names_averaged['features'].append(iaf_region_key)
            if paf_region_key not in features_dict and iaf_region_key not in features_dict:
                features_dict[paf_region_key] = [np.mean(curr_region_paf)]
                features_dict[iaf_region_key] = [np.mean(curr_region_iaf)]
            else:
                features_dict[paf_region_key].append(np.mean(curr_region_paf))
                features_dict[iaf_region_key].append(np.mean(curr_region_iaf))

        if 'class' not in features_dict:
            features_dict['class'] = [movement]
        else:
            features_dict['class'].append(movement)
        if 'class_simp' not in features_dict:
            if '1st' in data_file and 'im' in movement:
                features_dict['class_simp'] = [movement[:-1]]
            else:
                features_dict['class_simp'] = [movement]
        else:
            if '1st' in data_file and 'im' in movement:
                features_dict['class_simp'].append(movement[:-1])
            else:
                features_dict['class_simp'].append(movement)
        if 'subject' not in features_dict:
            features_dict['subject'] = [curr_index.split('_')[0]]
        else:
            features_dict['subject'].append(curr_index.split('_')[0])

features_names_all_df = pd.DataFrame.from_dict(features_dict_names_all)
features_names_all_df.to_excel(f"{dataframe_path}/features.xlsx", header=True, index=False)

features_names_averaged_df = pd.DataFrame.from_dict(features_dict_names_averaged)
features_names_averaged_df.to_excel(f"{dataframe_path}/features_region.xlsx", header=True, index=False)

features_names_alpha_beta_df = pd.DataFrame.from_dict(features_dict_names_alpha_beta)
features_names_alpha_beta_df.to_excel(f"{dataframe_path}/features_alpha_beta.xlsx", header=True, index=False)

features_names_alpha_df = pd.DataFrame.from_dict(features_dict_names_alpha)
features_names_alpha_df.to_excel(f"{dataframe_path}/features_alpha.xlsx", header=True, index=False)

features_df = pd.DataFrame.from_dict(features_dict)
features_df = shuffle(features_df)
features_df.reset_index(inplace=True, drop=True)
features_df.to_excel(f"{dataframe_path}/data.xlsx", header=True, index=False)

with open(f"{dataframe_path}/electrode_region_dict.pkl", 'wb') as f:
    pickle.dump(electrode_region_dict, f)
