import numpy as np
import mne
from mne.externals.pymatreader import read_mat
import matplotlib.pyplot as plt
from mne.baseline import rescale
from mne.stats import bootstrap_confidence_interval
from sklearn.decomposition import PCA, FastICA

data_path = 'E:/YandexDisk/EEG/'
data_file = 'preproc_data.mat'

mat_data = read_mat(data_path + data_file)
data_info = mne.create_info(ch_names=mat_data['res'][0]['right_real']['label'],
                            sfreq=mat_data['res'][0]['right_real']['fsample'],
                            ch_types='eeg')


def calculate_num_epochs(data, move_type):
    num_epochs = 0
    min_epoch_len = data['res'][0][move_type]['trial'][0].shape[1]
    for subject_id in range(0, len(data['res'])):
        if len(data['res'][subject_id]) > 0:
            num_epochs += len(data['res'][0][move_type]['trial'])
            for epoch_id in range(0, len(data['res'][0][move_type]['trial'])):
                if data['res'][subject_id][move_type]['trial'][epoch_id].shape[1] < min_epoch_len:
                    min_epoch_len = data['res'][subject_id][move_type]['trial'][epoch_id].shape[1]
    return num_epochs, min_epoch_len


num_epochs_right_real, min_epoch_len_right_real = calculate_num_epochs(mat_data, 'right_real')
num_epochs_right_quasi, min_epoch_len_right_quasi = calculate_num_epochs(mat_data, 'right_quasi')
num_epochs_right_im1, min_epoch_len_right_im1 = calculate_num_epochs(mat_data, 'right_im1')
num_epochs_right_im2, min_epoch_len_right_im2 = calculate_num_epochs(mat_data, 'right_im2')

num_epochs = min(num_epochs_right_real, num_epochs_right_quasi, num_epochs_right_im1, num_epochs_right_im2)
min_epoch_len = min(min_epoch_len_right_real, min_epoch_len_right_quasi,
                    min_epoch_len_right_im1, min_epoch_len_right_im2)

time_right_real = mat_data['res'][0]['right_real']['trial'][0].shape[0]
time_right_quasi = mat_data['res'][0]['right_quasi']['trial'][0].shape[0]
time_right_im1 = mat_data['res'][0]['right_im1']['trial'][0].shape[0]
time_right_im2 = mat_data['res'][0]['right_im2']['trial'][0].shape[0]
min_time = min(time_right_real, time_right_quasi, time_right_im1, time_right_im2)

data_right_real = np.empty(shape=(num_epochs, min_time, min_epoch_len))
data_right_quasi = np.empty(shape=(num_epochs, min_time, min_epoch_len))
data_right_im1 = np.empty(shape=(num_epochs, min_time, min_epoch_len))
data_right_im2 = np.empty(shape=(num_epochs, min_time, min_epoch_len))


def fill_data(data, raw_data, move_type):
    curr_epoch = 0
    for subject_id in range(0, len(raw_data['res'])):
        if len(raw_data['res'][subject_id]) > 0:
            for epoch_id in range(0, len(raw_data['res'][0][move_type]['trial'])):
                min_epoch_len = data.shape[2]
                data[curr_epoch, :, :] = raw_data['res'][subject_id][move_type]['trial'][epoch_id][:, - min_epoch_len:]
                curr_epoch += 1
    return data


data_right_real = fill_data(data_right_real, mat_data, 'right_real')
data_right_quasi = fill_data(data_right_quasi, mat_data, 'right_quasi')
data_right_im1 = fill_data(data_right_im1, mat_data, 'right_im1')
data_right_im2 = fill_data(data_right_im2, mat_data, 'right_im2')

freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]


def get_band_features(data, bands):
    band_features = np.empty(shape=(data.shape[0], data.shape[1] * 4))
    band_features_names = list()
    for band, f_min, f_max in bands:
        filtered_epochs = mne.EpochsArray(data=data.copy(), info=data_info)
        filtered_epochs.filter(f_min, f_max, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
        filtered_epochs.subtract_evoked()
        filtered_epochs.apply_hilbert(envelope=True)
        filtered_data = filtered_epochs.get_data()
        for lead_id in range(0, filtered_data.shape[1]):
            curr_lead = filtered_epochs.ch_names[lead_id]
            band_features_names.append('_'.join([band, 'mean', curr_lead]))
            band_features_names.append('_'.join([band, 'std', curr_lead]))
            band_features_names.append('_'.join([band, 'max', curr_lead]))
            band_features_names.append('_'.join([band, 'min', curr_lead]))
            for epoch_id in range(0, filtered_data.shape[0]):
                band_features[epoch_id, 4 * lead_id] = np.mean(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, 4 * lead_id + 1] = np.std(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, 4 * lead_id + 2] = np.max(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, 4 * lead_id + 3] = np.min(filtered_data[epoch_id, lead_id, :])
        del filtered_epochs
    return band_features, band_features_names

'''
band_features_right_real, band_features_names_right_real = get_band_features(data_right_real, freq_bands)
band_features_right_quasi, band_features_names_right_quasi = get_band_features(data_right_quasi, freq_bands)
band_features_right_im1, band_features_names_right_im1 = get_band_features(data_right_im1, freq_bands)
band_features_right_im2, band_features_names_right_im2 = get_band_features(data_right_im2, freq_bands)
'''
spec_bands = [('Spec', 1, 45)]


def get_spec_features(data, data_info, bands):
    spec_features = np.empty(shape=(data.shape[0], data.shape[1]))
    spec_features_names = list()
    for band, f_min, f_max in bands:
        spec_features_names.extend([band + lead for lead in data_info.ch_names])
        ssd = mne.decoding.SSD(info=data_info,
                               reg='oas',
                               filt_params_signal=dict(l_freq=f_min, h_freq=f_max,
                                                       l_trans_bandwidth=1, h_trans_bandwidth=1),
                               filt_params_noise=dict(l_freq=f_min - 1, h_freq=f_max + 1,
                                                      l_trans_bandwidth=1, h_trans_bandwidth=1))
        for epoch_id in range(0, data.shape[0]):
            ssd.fit(X=data[epoch_id, :, :].copy())
            ssd_sources = ssd.transform(X=data.copy())
            spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)
            spec_features[epoch_id, :] = spec_ratio
    return spec_features, spec_features_names


spec_features_right_real, spec_features_names_right_real = get_spec_features(data_right_real, data_info, spec_bands)
spec_features_right_quasi, spec_features_names_right_quasi = get_spec_features(data_right_quasi, data_info, spec_bands)
