import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter, periodogram
from scipy import fftpack
from sklearn.utils import shuffle
from mne.externals.pymatreader import read_mat
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

brain_regions_dict = {'Frontal': ['Fp1', 'Fp2', 'F3', 'Fz', 'F4'],
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

movements = list(data.keys())
freq_bands = [('Alpha', 8, 12), ('Beta', 12, 30)]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


features_dict = {}
features_names_dict = {'features': []}
for movement in data:
    print(movement)
    for trial_id in tqdm(range(0, len(data[movement]))):
        curr_index = data_indexes[movement][trial_id]
        if 'index' not in features_dict:
            features_dict['index'] = [curr_index]
        else:
            features_dict['index'].append(curr_index)
        curr_data = data[movement][trial_id]
        denoised_data = butter_bandpass_filter(curr_data, 1.0, 50.0, sample_frequency)

        for freq_band in freq_bands:
            band_name = freq_band[0]
            band_low = freq_band[1]
            band_high = freq_band[2]
            filtered_data = butter_bandpass_filter(denoised_data, band_low, band_high, sample_frequency)

            for region in brain_regions_dict:
                curr_region_psd = []
                curr_region_trp = []

                for electrode_id in range(0, len(brain_regions_dict[region])):
                    electrode_name = brain_regions_dict[region][electrode_id]

                    freqs, psd = periodogram(denoised_data[electrode_id, 5500:9500], sample_frequency)
                    ind_min = np.argmax(freqs > band_low) - 1
                    ind_max = np.argmax(freqs > band_high) - 1
                    curr_region_psd.append(np.trapz(psd[ind_min:ind_max], freqs[ind_min:ind_max]))

                    ps_rest = np.abs(np.fft.fft(denoised_data[electrode_id, 500:4500]))
                    freqs_rest = np.fft.fftfreq(denoised_data[electrode_id, 500:4500].size, 1 / sample_frequency)
                    ps_rest[np.abs(freqs_rest) < band_low] = 0
                    ps_rest[np.abs(freqs_rest) > band_high] = 0
                    filtered_rest = np.abs(fftpack.ifft(ps_rest)) ** 2
                    pow_rest = np.average(filtered_rest)

                    ps_act = np.abs(np.fft.fft(denoised_data[electrode_id, 5500:9500]))
                    freqs_act = np.fft.fftfreq(denoised_data[electrode_id, 5500:9500].size, 1 / sample_frequency)
                    ps_act[np.abs(freqs_act) < band_low] = 0
                    ps_act[np.abs(freqs_act) > band_high] = 0
                    filtered_act = np.abs(fftpack.ifft(ps_act)) ** 2
                    pow_act = np.average(filtered_act)
                    curr_region_trp.append(np.abs(np.log(pow_act) - np.log(pow_rest)))

                psd_key = f"{region}_{band_name}_PSD"
                if trial_id == 0 and movement == movements[0]:
                    features_names_dict['features'].append(psd_key)
                if psd_key not in features_dict:
                    features_dict[psd_key] = [np.mean(curr_region_psd)]
                else:
                    features_dict[psd_key].append(np.mean(curr_region_psd))

                trp_key = f"{region}_{band_name}_TRP"
                if trial_id == 0 and movement == movements[0]:
                    features_names_dict['features'].append(trp_key)
                if trp_key not in features_dict:
                    features_dict[trp_key] = [np.mean(curr_region_trp)]
                else:
                    features_dict[trp_key].append(np.mean(curr_region_trp))

        for region in brain_regions_dict:
            curr_region_paf = []
            curr_region_iaf = []

            for electrode_id in range(0, len(brain_regions_dict[region])):
                electrode_name = brain_regions_dict[region][electrode_id]

                ps_act = np.abs(np.fft.fft(denoised_data[electrode_id, 5500:9500])) ** 2
                freqs_act = np.fft.fftfreq(denoised_data[electrode_id, 5500:9500].size, 1 / sample_frequency)
                idx = np.argsort(freqs_act)
                ps = ps_act[idx]
                freqs = freqs_act[idx]
                alpha_freqs_ids = [freq_id for freq_id, freq in enumerate(list(freqs)) if (freq > 7.0 and freq < 13.0)]
                curr_region_paf.append(freqs[alpha_freqs_ids[np.argmax(ps[alpha_freqs_ids])]])
                curr_region_iaf.append(np.average(freqs[alpha_freqs_ids], weights=ps[alpha_freqs_ids]))

            paf_key = f"{region}_PAF"
            if trial_id == 0 and movement == movements[0]:
                features_names_dict['features'].append(paf_key)
            if paf_key not in features_dict:
                features_dict[paf_key] = [np.mean(curr_region_paf)]
            else:
                features_dict[paf_key].append(np.mean(curr_region_paf))

            iaf_key = f"{region}_IAF"
            if trial_id == 0 and movement == movements[0]:
                features_names_dict['features'].append(iaf_key)
            if iaf_key not in features_dict:
                features_dict[iaf_key] = [np.mean(curr_region_iaf)]
            else:
                features_dict[iaf_key].append(np.mean(curr_region_iaf))

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

features_names_df = pd.DataFrame.from_dict(features_names_dict)
features_names_df.to_excel(f"{dataframe_path}/features.xlsx", header=True, index=False)

features_df = pd.DataFrame.from_dict(features_dict)
features_df = shuffle(features_df)
features_df.reset_index(inplace=True, drop=True)
features_df.to_excel(f"{dataframe_path}/data.xlsx", header=True, index=False)

with open(f"{dataframe_path}/electrode_region_dict.pkl", 'wb') as f:
    pickle.dump(electrode_region_dict, f)
