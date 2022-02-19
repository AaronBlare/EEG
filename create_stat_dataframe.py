import numpy as np
import mne
from mne.externals.pymatreader import read_mat
from mne.time_frequency import tfr_multitaper
import pandas as pd


data_path = 'E:/YandexDisk/EEG/'
data_file = 'preproc_data.mat'

mat_data = read_mat(data_path + data_file)
data_info = mne.create_info(ch_names=mat_data['res'][0]['right_real']['label'],
                            sfreq=mat_data['res'][0]['right_real']['fsample'],
                            ch_types='eeg')


def calculate_num_trials(data, move_type):
    num_trials = 0
    min_trial_len = data['res'][0][move_type]['trial'][0].shape[1]
    for subject_id in range(0, len(data['res'])):
        if len(data['res'][subject_id]) > 0:
            num_trials += len(data['res'][0][move_type]['trial'])
            for epoch_id in range(0, len(data['res'][0][move_type]['trial'])):
                if data['res'][subject_id][move_type]['trial'][epoch_id].shape[1] < min_trial_len:
                    min_trial_len = data['res'][subject_id][move_type]['trial'][epoch_id].shape[1]
    return num_trials, min_trial_len


num_trials_right_real, min_trial_len_right_real = calculate_num_trials(mat_data, 'right_real')
num_trials_right_quasi, min_trial_len_right_quasi = calculate_num_trials(mat_data, 'right_quasi')
num_trials_right_im1, min_trial_len_right_im1 = calculate_num_trials(mat_data, 'right_im1')
num_trials_right_im2, min_trial_len_right_im2 = calculate_num_trials(mat_data, 'right_im2')

num_trials = min(num_trials_right_real, num_trials_right_quasi, num_trials_right_im1, num_trials_right_im2)
min_trial_len = min(min_trial_len_right_real, min_trial_len_right_quasi,
                    min_trial_len_right_im1, min_trial_len_right_im2)

time_right_real = mat_data['res'][0]['right_real']['trial'][0].shape[0]
time_right_quasi = mat_data['res'][0]['right_quasi']['trial'][0].shape[0]
time_right_im1 = mat_data['res'][0]['right_im1']['trial'][0].shape[0]
time_right_im2 = mat_data['res'][0]['right_im2']['trial'][0].shape[0]
min_time = min(time_right_real, time_right_quasi, time_right_im1, time_right_im2)

data_right_real = np.empty(shape=(num_trials, min_time, min_trial_len))
data_right_quasi = np.empty(shape=(num_trials, min_time, min_trial_len))
data_right_im1 = np.empty(shape=(num_trials, min_time, min_trial_len))
data_right_im2 = np.empty(shape=(num_trials, min_time, min_trial_len))


def fill_data(data, raw_data, move_type, trial_len):
    curr_trial = 0
    for subject_id in range(0, len(raw_data['res'])):
        if len(raw_data['res'][subject_id]) > 0:
            for trial_id in range(0, len(raw_data['res'][0][move_type]['trial'])):
                data[curr_trial, :, :] = raw_data['res'][subject_id][move_type]['trial'][trial_id][:, :trial_len]
                curr_trial += 1
    return data


data_right_real = fill_data(data_right_real, mat_data, 'right_real', min_trial_len)
data_right_quasi = fill_data(data_right_quasi, mat_data, 'right_quasi', min_trial_len)
data_right_im1 = fill_data(data_right_im1, mat_data, 'right_im1', min_trial_len)
data_right_im2 = fill_data(data_right_im2, mat_data, 'right_im2', min_trial_len)

all_data = np.concatenate((data_right_real, data_right_quasi, data_right_im1, data_right_im2), axis=0)

freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]


def get_band_features(data, bands):
    band_features = np.empty(shape=(data.shape[0], data.shape[1] * 5 * 5))
    band_features_names = list()
    band_id = 0
    epochs = mne.EpochsArray(data=data[:, :, :5000].copy(), info=data_info)
    tfr = tfr_multitaper(epochs, freqs=np.arange(2, 36), n_cycles=np.arange(2, 36), use_fft=True, return_itc=False,
                         average=False, decim=2)
    for band, f_min, f_max in bands:
        filtered_epochs = mne.EpochsArray(data=data.copy(), info=data_info)
        filtered_epochs.filter(1, 50, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
        filtered_epochs.filter(f_min, f_max, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
        filtered_data = filtered_epochs.get_data()
        for lead_id in range(0, filtered_data.shape[1]):
            curr_lead = filtered_epochs.ch_names[lead_id]
            band_features_names.append('_'.join([band, 'mean', curr_lead]))
            band_features_names.append('_'.join([band, 'median', curr_lead]))
            band_features_names.append('_'.join([band, 'std', curr_lead]))
            band_features_names.append('_'.join([band, 'max', curr_lead]))
            band_features_names.append('_'.join([band, 'min', curr_lead]))
            for epoch_id in range(0, filtered_data.shape[0]):
                band_features[epoch_id, (band_id + 1) * 4 * lead_id] = np.mean(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, (band_id + 1) * 4 * lead_id] = np.median(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, (band_id + 1) * 4 * lead_id + 1] = np.std(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, (band_id + 1) * 4 * lead_id + 2] = np.max(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, (band_id + 1) * 4 * lead_id + 3] = np.min(filtered_data[epoch_id, lead_id, :])
        del filtered_epochs
        band_id += 1
    return band_features, band_features_names


band_features_right_real, band_features_names_right_real = get_band_features(data_right_real, freq_bands)
band_features_right_quasi, band_features_names_right_quasi = get_band_features(data_right_quasi, freq_bands)
band_features_right_im1, band_features_names_right_im1 = get_band_features(data_right_im1, freq_bands)
band_features_right_im2, band_features_names_right_im2 = get_band_features(data_right_im2, freq_bands)

all_features = np.concatenate((band_features_right_real, band_features_right_quasi,
                               band_features_right_im1, band_features_right_im2), axis=0)
all_names = band_features_names_right_real
all_classes = [0] * band_features_right_real.shape[0] + [1] * band_features_right_quasi.shape[0] + \
              [2] * band_features_right_im1.shape[0] + [3] * band_features_right_im2.shape[0]

all_classes_names = ['real'] * band_features_right_real.shape[0] + ['quasi'] * band_features_right_quasi.shape[0] + \
                    ['im1'] * band_features_right_im1.shape[0] + ['im2'] * band_features_right_im2.shape[0]

df = pd.DataFrame(np.concatenate((all_features, np.c_[all_classes_names]), axis=1),
                  columns=all_names + ['class'])
df.to_excel("dataframes/dataframe.xlsx", header=True, index=False)
