import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, lfilter, welch
from scipy import fftpack
from sklearn.utils import shuffle
from mne.externals.pymatreader import read_mat
from tqdm import tqdm


data_path = 'E:/YandexDisk/EEG/raw/'

data_files = ['1st_Day.mat']
dataframe_path = 'E:/YandexDisk/EEG/Dataframes/'

amplitude_characteristics = True

data = {}
data_indexes = {}
for file_id in range(0, len(data_files)):
    start_index = 0
    curr_mat_data = read_mat(data_path + data_files[file_id])
    if file_id == 0:
        if 'subs_ica' in curr_mat_data:
            electrodes_names = curr_mat_data['subs_ica'][0]['right_real']['label']
            sample_frequency = curr_mat_data['subs_ica'][0]['right_real']['fsample']
            for subject_id in range(0, len(curr_mat_data['subs_ica'])):
                for key in curr_mat_data['subs_ica'][subject_id]:
                    if key not in data:
                        data[key] = []
                        data_indexes[key] = []
                    for trial_id in range(0, len(curr_mat_data['subs_ica'][subject_id][key]['trial'])):
                        data[key].append(curr_mat_data['subs_ica'][subject_id][key]['trial'][trial_id] * 10 ** 6)
                        data_indexes[key].append(f"S{start_index + subject_id}_T{trial_id}_{key}")
            start_index += len(curr_mat_data['subs_ica'])
        else:
            electrodes_names = curr_mat_data['res'][0]['right_real']['label']
            sample_frequency = curr_mat_data['res'][0]['right_real']['fsample']
            for subject_id in range(0, len(curr_mat_data['res'])):
                for key in curr_mat_data['res'][subject_id]:
                    if key not in data:
                        data[key] = []
                        data_indexes[key] = []
                    for trial_id in range(0, len(curr_mat_data['res'][subject_id][key]['trial'])):
                        data[key].append(curr_mat_data['res'][subject_id][key]['trial'][trial_id] * 10 ** 6)
                        data_indexes[key].append(f"S{start_index + subject_id}_T{trial_id}_{key}")
            start_index += len(curr_mat_data['res'])
    del curr_mat_data

movements = list(data.keys())
freq_bands = [('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]


def butter_bandpass(lowcut, highcut, fs, order=5):
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
features_names_dict = {'features': []}
features_names_freq_dict = {'features': []}
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

        for electrode_id in range(0, len(electrodes_names)):
            electrode_name = electrodes_names[electrode_id]

            for freq_band in freq_bands:
                band_name = freq_band[0]
                band_low = freq_band[1]
                band_high = freq_band[2]
                filtered_data = butter_bandpass_filter(denoised_data, band_low, band_high, sample_frequency)

                if amplitude_characteristics:

                    mean_key = f"{electrode_name}_{band_name}_mean"
                    if trial_id == 0 and movement == movements[0]:
                        features_names_dict['features'].append(mean_key)
                    if mean_key not in features_dict:
                        features_dict[mean_key] = [np.mean(filtered_data[electrode_id, 5500:9500])]
                    else:
                        features_dict[mean_key].append(np.mean(filtered_data[electrode_id, 5500:9500]))

                    median_key = f"{electrode_name}_{band_name}_median"
                    if trial_id == 0 and movement == movements[0]:
                        features_names_dict['features'].append(median_key)
                    if median_key not in features_dict:
                        features_dict[median_key] = [np.median(filtered_data[electrode_id, 5500:9500])]
                    else:
                        features_dict[median_key].append(np.median(filtered_data[electrode_id, 5500:9500]))

                    std_key = f"{electrode_name}_{band_name}_std"
                    if trial_id == 0 and movement == movements[0]:
                        features_names_dict['features'].append(std_key)
                    if std_key not in features_dict:
                        features_dict[std_key] = [np.std(filtered_data[electrode_id, 5500:9500])]
                    else:
                        features_dict[std_key].append(np.std(filtered_data[electrode_id, 5500:9500]))

                    min_key = f"{electrode_name}_{band_name}_min"
                    if trial_id == 0 and movement == movements[0]:
                        features_names_dict['features'].append(min_key)
                    if min_key not in features_dict:
                        features_dict[min_key] = [np.min(filtered_data[electrode_id, 5500:9500])]
                    else:
                        features_dict[min_key].append(np.min(filtered_data[electrode_id, 5500:9500]))

                    max_key = f"{electrode_name}_{band_name}_max"
                    if trial_id == 0 and movement == movements[0]:
                        features_names_dict['features'].append(max_key)
                    if max_key not in features_dict:
                        features_dict[max_key] = [np.max(filtered_data[electrode_id, 5500:9500])]
                    else:
                        features_dict[max_key].append(np.max(filtered_data[electrode_id, 5500:9500]))

                freqs, psd = welch(denoised_data[electrode_id, 5500:9500], sample_frequency)
                curr_freqs_ids = [freq_id for freq_id, freq in enumerate(list(freqs)) if
                                  (freq > band_low and freq < band_high)]
                psd_key = f"{electrode_name}_{band_name}_psd"
                if trial_id == 0 and movement == movements[0]:
                    features_names_dict['features'].append(psd_key)
                    features_names_freq_dict['features'].append(psd_key)
                if psd_key not in features_dict:
                    features_dict[psd_key] = [np.mean(psd[curr_freqs_ids])]
                else:
                    features_dict[psd_key].append(np.mean(psd[curr_freqs_ids]))

                ps_rest = np.abs(np.fft.fft(denoised_data[electrode_id, 0:4000]))
                freqs_rest = np.fft.fftfreq(denoised_data[electrode_id, 0:4000].size, 1 / sample_frequency)
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

                trp_key = f"{electrode_name}_{band_name}_trp"
                if trial_id == 0 and movement == movements[0]:
                    features_names_dict['features'].append(trp_key)
                    features_names_freq_dict['features'].append(trp_key)
                if trp_key not in features_dict:
                    features_dict[trp_key] = [np.log(pow_act) - np.log(pow_rest)]
                else:
                    features_dict[trp_key].append(np.log(pow_act) - np.log(pow_rest))

            ps_act = np.abs(np.fft.fft(denoised_data[electrode_id, 5500:9500])) ** 2
            freqs_act = np.fft.fftfreq(denoised_data[electrode_id, 5500:9500].size, 1 / sample_frequency)
            idx = np.argsort(freqs_act)
            ps = ps_act[idx]
            freqs = freqs_act[idx]
            alpha_freqs_ids = [freq_id for freq_id, freq in enumerate(list(freqs)) if (freq > 7.0 and freq < 13.0)]
            paf_key = f"{electrode_name}_paf"
            if trial_id == 0 and movement == movements[0]:
                features_names_dict['features'].append(paf_key)
                features_names_freq_dict['features'].append(paf_key)
            if paf_key not in features_dict:
                features_dict[paf_key] = [freqs[alpha_freqs_ids[np.argmax(ps[alpha_freqs_ids])]]]
            else:
                features_dict[paf_key].append(freqs[alpha_freqs_ids[np.argmax(ps[alpha_freqs_ids])]])

            iaf_key = f"{electrode_name}_iaf"
            if trial_id == 0 and movement == movements[0]:
                features_names_dict['features'].append(iaf_key)
                features_names_freq_dict['features'].append(iaf_key)
            if iaf_key not in features_dict:
                features_dict[iaf_key] = [np.average(freqs[alpha_freqs_ids], weights=ps[alpha_freqs_ids])]
            else:
                features_dict[iaf_key].append(np.average(freqs[alpha_freqs_ids], weights=ps[alpha_freqs_ids]))

        if 'class' not in features_dict:
            features_dict['class'] = [movement]
        else:
            features_dict['class'].append(movement)
        if 'class_simp' not in features_dict:
            if '1st' in data_files[0] and 'im' in movement:
                features_dict['class_simp'] = [movement[:-1]]
            else:
                features_dict['class_simp'] = [movement]
        else:
            if '1st' in data_files[0] and 'im' in movement:
                features_dict['class_simp'].append(movement[:-1])
            else:
                features_dict['class_simp'].append(movement)
        if 'subject' not in features_dict:
            features_dict['subject'] = [curr_index.split('_')[0]]
        else:
            features_dict['subject'].append(curr_index.split('_')[0])

features_names_df = pd.DataFrame.from_dict(features_names_dict)
features_names_df.to_excel(f"{dataframe_path}/features.xlsx", header=True, index=False)
features_names_freq_df = pd.DataFrame.from_dict(features_names_freq_dict)
features_names_freq_df.to_excel(f"{dataframe_path}/features_freq.xlsx", header=True, index=False)

features_df = pd.DataFrame.from_dict(features_dict)
features_df = shuffle(features_df)
features_df.reset_index(inplace=True, drop=True)
features_df.to_excel(f"{dataframe_path}/data.xlsx", header=True, index=False)
