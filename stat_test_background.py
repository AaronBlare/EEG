import numpy as np
import mne
from mne.externals.pymatreader import read_mat
import pandas as pd
import scipy
from scipy.stats import kruskal, mannwhitneyu
import statsmodels
from statsmodels.stats.multitest import multipletests

data_path = 'E:/YandexDisk/EEG/Data/'
old_data_file = '2nd_Day_TMS.mat'
new_data_files = ['1st_Day_background_recordings.mat', '2nd_Day_sham_group.mat', '2nd_Day_TMS.mat',
                  '2nd_Day_TMS_5new_subj.mat']


def calculate_num_trials(data, move_type):
    if 'subs_ica_bgr' in data:
        dict_flag = 'subs_ica_bgr'
    else:
        dict_flag = 'res_bgr'
    num_trials = 0
    min_trial_len = data[dict_flag][0][move_type]['trial'].shape[1]
    for subject_id in range(0, len(data[dict_flag])):
        if len(data[dict_flag][subject_id]) > 0:
            num_trials += 1
            if data[dict_flag][subject_id][move_type]['trial'].shape[1] < min_trial_len:
                min_trial_len = data[dict_flag][subject_id][move_type]['trial'].shape[1]
    return num_trials, min_trial_len


def fill_data(data, raw_data, move_type, trial_len):
    if 'subs_ica_bgr' in raw_data:
        dict_flag = 'subs_ica_bgr'
    else:
        dict_flag = 'res_bgr'
    curr_trial = 0
    for subject_id in range(0, len(raw_data[dict_flag])):
        if len(raw_data[dict_flag][subject_id]) > 0:
            data[curr_trial, :, :] = raw_data[dict_flag][subject_id][move_type]['trial'][:, :trial_len]
            curr_trial += 1
    return data


def get_band_features(data, bands):
    band_features = np.empty(shape=(data.shape[0], data.shape[1] * 5 * 5))
    band_features_names = list()
    band_id = 0
    feature_id = 0
    for band, f_min, f_max in bands:
        filtered_epochs = mne.EpochsArray(data=data.copy(), info=data_info)
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
                band_features[epoch_id, feature_id] = np.mean(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, feature_id + 1] = np.median(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, feature_id + 2] = np.std(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, feature_id + 3] = np.max(filtered_data[epoch_id, lead_id, :])
                band_features[epoch_id, feature_id + 4] = np.min(filtered_data[epoch_id, lead_id, :])
            feature_id += 5
        del filtered_epochs
        band_id += 1
    return band_features, band_features_names


calculate_old_background = False
calculate_1st_background = False
calculate_all_background = False
calculate_all_background_by_class = False
calculate_all_background_by_class_merged = False

if calculate_old_background:

    old_mat_data = read_mat(data_path + old_data_file)
    data_info = mne.create_info(ch_names=old_mat_data['res_bgr'][0]['right_real']['label'],
                                sfreq=old_mat_data['res_bgr'][0]['right_real']['fsample'],
                                ch_types='eeg')

    num_trials_real, min_trial_len_real = calculate_num_trials(old_mat_data, 'right_real')
    num_trials_quasi, min_trial_len_quasi = calculate_num_trials(old_mat_data, 'right_quasi')
    num_trials_im1, min_trial_len_im1 = calculate_num_trials(old_mat_data, 'right_im1')
    num_trials_im2, min_trial_len_im2 = calculate_num_trials(old_mat_data, 'right_im2')

    num_trials = min(num_trials_real, num_trials_quasi, num_trials_im1, num_trials_im2)
    min_trial_len = min(min_trial_len_real, min_trial_len_quasi, min_trial_len_im1, min_trial_len_im2)

    num_electrodes = old_mat_data['res_bgr'][0]['right_real']['trial'].shape[0]

    data_real = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_quasi = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im1 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im2 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))

    data_real = fill_data(data_real, old_mat_data, 'right_real', min_trial_len)
    data_quasi = fill_data(data_quasi, old_mat_data, 'right_quasi', min_trial_len)
    data_im1 = fill_data(data_im1, old_mat_data, 'right_im1', min_trial_len)
    data_im2 = fill_data(data_im2, old_mat_data, 'right_im2', min_trial_len)

    freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

    band_features_real, band_features_names_real = get_band_features(data_real, freq_bands)
    band_features_quasi, band_features_names_quasi = get_band_features(data_quasi, freq_bands)
    band_features_im1, band_features_names_im1 = get_band_features(data_im1, freq_bands)
    band_features_im2, band_features_names_im2 = get_band_features(data_im2, freq_bands)

    kruskal_pval_old_background = {'feature': [], 'pval': [], 'pval_bh': []}
    for feature_id in range(0, len(band_features_names_real)):
        if len(set(band_features_real[:, feature_id] + band_features_quasi[:, feature_id] +
                   band_features_im1[:, feature_id] + band_features_im2[:, feature_id])) > 1:
            stat, pval = kruskal(band_features_real[:, feature_id], band_features_quasi[:, feature_id],
                                 band_features_im1[:, feature_id], band_features_im2[:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_old_background['pval'].append(pval)
                kruskal_pval_old_background['feature'].append(band_features_names_real[feature_id])

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_old_background['pval'], method='fdr_bh')
    kruskal_pval_old_background['pval_bh'] = pval_bh

    df = pd.DataFrame.from_dict(kruskal_pval_old_background)
    df.to_excel("files/kruskal_old_background.xlsx", header=True, index=False)

if calculate_1st_background:

    mat_data = read_mat(data_path + new_data_files[0])
    data_info = mne.create_info(ch_names=mat_data['subs_ica_bgr'][0]['right_real']['label'],
                                sfreq=mat_data['subs_ica_bgr'][0]['right_real']['fsample'],
                                ch_types='eeg')

    num_trials_real, min_trial_len_real = calculate_num_trials(mat_data, 'right_real')
    num_trials_quasi, min_trial_len_quasi = calculate_num_trials(mat_data, 'right_quasi')
    num_trials_im1, min_trial_len_im1 = calculate_num_trials(mat_data, 'right_im1')
    num_trials_im2, min_trial_len_im2 = calculate_num_trials(mat_data, 'right_im2')

    num_trials = min(num_trials_real, num_trials_quasi, num_trials_im1, num_trials_im2)
    min_trial_len = min(min_trial_len_real, min_trial_len_quasi, min_trial_len_im1, min_trial_len_im2)

    num_electrodes = mat_data['subs_ica_bgr'][0]['right_real']['trial'].shape[0]

    data_real = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_quasi = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im1 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im2 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))

    data_real = fill_data(data_real, mat_data, 'right_real', min_trial_len)
    data_quasi = fill_data(data_quasi, mat_data, 'right_quasi', min_trial_len)
    data_im1 = fill_data(data_im1, mat_data, 'right_im1', min_trial_len)
    data_im2 = fill_data(data_im2, mat_data, 'right_im2', min_trial_len)

    freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

    band_features_real, band_features_names_real = get_band_features(data_real, freq_bands)
    band_features_quasi, band_features_names_quasi = get_band_features(data_quasi, freq_bands)
    band_features_im1, band_features_names_im1 = get_band_features(data_im1, freq_bands)
    band_features_im2, band_features_names_im2 = get_band_features(data_im2, freq_bands)

    kruskal_pval_left_right_background = {'feature': [], 'pval': [], 'pval_bh': []}
    for feature_id in range(0, len(band_features_names_real)):
        if len(set(band_features_real[:, feature_id] + band_features_quasi[:, feature_id] +
                   band_features_im1[:, feature_id] + band_features_im2[:, feature_id])) > 1:
            stat, pval = kruskal(band_features_real[:, feature_id], band_features_quasi[:, feature_id],
                                 band_features_im1[:, feature_id], band_features_im2[:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_left_right_background['pval'].append(pval)
                kruskal_pval_left_right_background['feature'].append(band_features_names_real[feature_id])

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_left_right_background['pval'], method='fdr_bh')
    kruskal_pval_left_right_background['pval_bh'] = pval_bh

    df = pd.DataFrame.from_dict(kruskal_pval_left_right_background)
    df.to_excel("files/kruskal_left_right_background.xlsx", header=True, index=False)

mat_data = read_mat(data_path + new_data_files[0])
data_info = mne.create_info(ch_names=mat_data['subs_ica_bgr'][0]['right_real']['label'],
                            sfreq=mat_data['subs_ica_bgr'][0]['right_real']['fsample'],
                            ch_types='eeg')

num_trials_right_im1, min_trial_len_right_im1 = calculate_num_trials(mat_data, 'right_im1')
num_trials_right_im2, min_trial_len_right_im2 = calculate_num_trials(mat_data, 'right_im2')

num_trials_left_im1, min_trial_len_left_im1 = calculate_num_trials(mat_data, 'left_im1')
num_trials_left_im2, min_trial_len_left_im2 = calculate_num_trials(mat_data, 'left_im2')

num_trials_right = min(num_trials_right_im1, num_trials_right_im2)
min_trial_len_right = min(min_trial_len_right_im1, min_trial_len_right_im2)

num_trials_left = min(num_trials_left_im1, num_trials_left_im2)
min_trial_len_left = min(min_trial_len_left_im1, min_trial_len_left_im2)

num_electrodes = mat_data['subs_ica_bgr'][0]['right_real']['trial'].shape[0]

data_right_im1 = np.empty(shape=(num_trials_right, num_electrodes, min_trial_len_right))
data_right_im2 = np.empty(shape=(num_trials_right, num_electrodes, min_trial_len_right))

data_left_im1 = np.empty(shape=(num_trials_left, num_electrodes, min_trial_len_left))
data_left_im2 = np.empty(shape=(num_trials_left, num_electrodes, min_trial_len_left))

data_right_im1 = fill_data(data_right_im1, mat_data, 'right_im1', min_trial_len_right)
data_right_im2 = fill_data(data_right_im2, mat_data, 'right_im2', min_trial_len_right)

data_left_im1 = fill_data(data_left_im1, mat_data, 'right_im1', min_trial_len_left)
data_left_im2 = fill_data(data_left_im2, mat_data, 'right_im2', min_trial_len_left)

freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

band_features_right_im1, band_features_names_right_im1 = get_band_features(data_right_im1, freq_bands)
band_features_right_im2, band_features_names_right_im2 = get_band_features(data_right_im2, freq_bands)

band_features_left_im1, band_features_names_left_im1 = get_band_features(data_left_im1, freq_bands)
band_features_left_im2, band_features_names_left_im2 = get_band_features(data_left_im2, freq_bands)

mannwhitney_pval_right_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
for feature_id in range(0, len(band_features_names_right_im1)):
    if len(set(band_features_right_im1[:, feature_id] + band_features_right_im2[:, feature_id])) > 1:
        res = mannwhitneyu(band_features_right_im1[:, feature_id], band_features_right_im2[:, feature_id])
        pval = res.pvalue
        if not np.isnan(pval):
            mannwhitney_pval_right_im1_im2['pval'].append(pval)
            mannwhitney_pval_right_im1_im2['feature'].append(band_features_names_right_im1[feature_id])

reject, pval_bh, alphacSidak, alphacBonf = multipletests(mannwhitney_pval_right_im1_im2['pval'], method='fdr_bh')
mannwhitney_pval_right_im1_im2['pval_bh'] = pval_bh

df = pd.DataFrame.from_dict(mannwhitney_pval_right_im1_im2)
df.to_excel("files/mannwhitney_right_im1_im2.xlsx", header=True, index=False)

mannwhitney_pval_left_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
for feature_id in range(0, len(band_features_names_left_im1)):
    if len(set(band_features_left_im1[:, feature_id] + band_features_left_im2[:, feature_id])) > 1:
        res = mannwhitneyu(band_features_left_im1[:, feature_id], band_features_left_im2[:, feature_id])
        pval = res.pvalue
        if not np.isnan(pval):
            mannwhitney_pval_left_im1_im2['pval'].append(pval)
            mannwhitney_pval_left_im1_im2['feature'].append(band_features_names_left_im1[feature_id])

reject, pval_bh, alphacSidak, alphacBonf = multipletests(mannwhitney_pval_left_im1_im2['pval'], method='fdr_bh')
mannwhitney_pval_left_im1_im2['pval_bh'] = pval_bh

df = pd.DataFrame.from_dict(mannwhitney_pval_left_im1_im2)
df.to_excel("files/mannwhitney_left_im1_im2.xlsx", header=True, index=False)

if calculate_all_background:

    mat_data = []
    num_trials_files = []
    min_trial_len_files = []

    for file_id in range(0, len(new_data_files)):

        file_name = new_data_files[file_id]
        mat_data.append(read_mat(data_path + file_name))

        if file_name.startswith('1st'):
            data_info = mne.create_info(ch_names=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['label'],
                                        sfreq=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['fsample'],
                                        ch_types='eeg')

        num_trials_real, min_trial_len_real = calculate_num_trials(mat_data[file_id], 'right_real')
        num_trials_quasi, min_trial_len_quasi = calculate_num_trials(mat_data[file_id], 'right_quasi')
        num_trials_im1, min_trial_len_im1 = calculate_num_trials(mat_data[file_id], 'right_im1')
        num_trials_im2, min_trial_len_im2 = calculate_num_trials(mat_data[file_id], 'right_im2')

        num_trials = min(num_trials_real, num_trials_quasi, num_trials_im1, num_trials_im2)
        min_trial_len = min(min_trial_len_real, min_trial_len_quasi, min_trial_len_im1, min_trial_len_im2)

        num_trials_files.append(num_trials)
        min_trial_len_files.append(min_trial_len)

    num_trials = sum(num_trials_files)
    min_trial_len = min(min_trial_len_files)
    num_electrodes = mat_data[0]['subs_ica_bgr'][0]['right_real']['trial'].shape[0]

    data_real = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_quasi = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im1 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))
    data_im2 = np.empty(shape=(num_trials, num_electrodes, min_trial_len))

    curr_num_trials = 0

    for file_id in range(0, len(new_data_files)):
        curr_data_real = np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len))
        curr_data_quasi = np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len))
        curr_data_im1 = np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len))
        curr_data_im2 = np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len))

        curr_data_real = fill_data(curr_data_real, mat_data[file_id], 'right_real', min_trial_len)
        curr_data_quasi = fill_data(curr_data_quasi, mat_data[file_id], 'right_quasi', min_trial_len)
        curr_data_im1 = fill_data(curr_data_im1, mat_data[file_id], 'right_im1', min_trial_len)
        curr_data_im2 = fill_data(curr_data_im2, mat_data[file_id], 'right_im2', min_trial_len)

        data_real[curr_num_trials:(curr_num_trials + num_trials_files[file_id]), :, :] = curr_data_real
        data_quasi[curr_num_trials:(curr_num_trials + num_trials_files[file_id]), :, :] = curr_data_quasi
        data_im1[curr_num_trials:(curr_num_trials + num_trials_files[file_id]), :, :] = curr_data_im1
        data_im2[curr_num_trials:(curr_num_trials + num_trials_files[file_id]), :, :] = curr_data_im2

        curr_num_trials += num_trials_files[file_id]

    freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

    band_features_real, band_features_names_real = get_band_features(data_real, freq_bands)
    band_features_quasi, band_features_names_quasi = get_band_features(data_quasi, freq_bands)
    band_features_im1, band_features_names_im1 = get_band_features(data_im1, freq_bands)
    band_features_im2, band_features_names_im2 = get_band_features(data_im2, freq_bands)

    kruskal_pval_all_background = {'feature': [], 'pval': [], 'pval_bh': []}
    for feature_id in range(0, len(band_features_names_real)):
        if len(set(band_features_real[:, feature_id] + band_features_quasi[:, feature_id] +
                   band_features_im1[:, feature_id] + band_features_im2[:, feature_id])) > 1:
            stat, pval = kruskal(band_features_real[:, feature_id], band_features_quasi[:, feature_id],
                                 band_features_im1[:, feature_id], band_features_im2[:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background['pval'].append(pval)
                kruskal_pval_all_background['feature'].append(band_features_names_real[feature_id])

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background['pval'], method='fdr_bh')
    kruskal_pval_all_background['pval_bh'] = pval_bh

    df = pd.DataFrame.from_dict(kruskal_pval_all_background)
    df.to_excel("files/kruskal_all_background.xlsx", header=True, index=False)

if calculate_all_background_by_class:
    mat_data = []
    num_trials_files = []
    min_trial_len_files = []

    for file_id in range(0, len(new_data_files)):

        file_name = new_data_files[file_id]
        mat_data.append(read_mat(data_path + file_name))

        if file_name.startswith('1st'):
            data_info = mne.create_info(ch_names=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['label'],
                                        sfreq=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['fsample'],
                                        ch_types='eeg')

        num_trials_real, min_trial_len_real = calculate_num_trials(mat_data[file_id], 'right_real')
        num_trials_quasi, min_trial_len_quasi = calculate_num_trials(mat_data[file_id], 'right_quasi')
        num_trials_im1, min_trial_len_im1 = calculate_num_trials(mat_data[file_id], 'right_im1')
        num_trials_im2, min_trial_len_im2 = calculate_num_trials(mat_data[file_id], 'right_im2')

        num_trials = min(num_trials_real, num_trials_quasi, num_trials_im1, num_trials_im2)
        min_trial_len = min(min_trial_len_real, min_trial_len_quasi, min_trial_len_im1, min_trial_len_im2)

        num_trials_files.append(num_trials)
        min_trial_len_files.append(min_trial_len)

    min_trial_len = min(min_trial_len_files)
    num_electrodes = mat_data[0]['subs_ica_bgr'][0]['right_real']['trial'].shape[0]

    data_real = []
    data_quasi = []
    data_im1 = []
    data_im2 = []

    for file_id in range(0, len(new_data_files)):
        data_real.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_quasi.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_im1.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_im2.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))

    freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

    band_features_real = []
    band_features_quasi = []
    band_features_im1 = []
    band_features_im2 = []

    for file_id in range(0, len(new_data_files)):
        data_real[file_id] = fill_data(data_real[file_id], mat_data[file_id], 'right_real', min_trial_len)
        data_quasi[file_id] = fill_data(data_quasi[file_id], mat_data[file_id], 'right_quasi', min_trial_len)
        data_im1[file_id] = fill_data(data_im1[file_id], mat_data[file_id], 'right_im1', min_trial_len)
        data_im2[file_id] = fill_data(data_im2[file_id], mat_data[file_id], 'right_im2', min_trial_len)

        curr_band_features_real, band_features_names_real = get_band_features(data_real[file_id], freq_bands)
        curr_band_features_quasi, band_features_names_quasi = get_band_features(data_quasi[file_id], freq_bands)
        curr_band_features_im1, band_features_names_im1 = get_band_features(data_im1[file_id], freq_bands)
        curr_band_features_im2, band_features_names_im2 = get_band_features(data_im2[file_id], freq_bands)

        band_features_real.append(curr_band_features_real)
        band_features_quasi.append(curr_band_features_quasi)
        band_features_im1.append(curr_band_features_im1)
        band_features_im2.append(curr_band_features_im2)

    kruskal_pval_all_background_by_file_real = {'feature_real': [],
                                                'pval_real': [], 'pval_bh_real': []}
    kruskal_pval_all_background_by_file_quasi = {'feature_quasi': [],
                                                 'pval_quasi': [], 'pval_bh_quasi': []}
    kruskal_pval_all_background_by_file_im1 = {'feature_im1': [],
                                               'pval_im1': [], 'pval_bh_im1': []}
    kruskal_pval_all_background_by_file_im2 = {'feature_im2': [],
                                               'pval_im2': [], 'pval_bh_im2': []}

    for feature_id in range(0, len(band_features_names_real)):
        if len(set(band_features_real[0][:, feature_id]).union(set(band_features_real[1][:, feature_id])).union(
                set(band_features_real[2][:, feature_id])).union(set(band_features_real[3][:, feature_id]))) > 1:
            stat, pval = kruskal(band_features_real[0][:, feature_id], band_features_real[1][:, feature_id],
                                 band_features_real[2][:, feature_id], band_features_real[3][:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background_by_file_real['pval_real'].append(pval)
                kruskal_pval_all_background_by_file_real['feature_real'].append(band_features_names_real[feature_id])

        if len(set(band_features_quasi[0][:, feature_id]).union(set(band_features_quasi[1][:, feature_id])).union(
                set(band_features_quasi[2][:, feature_id])).union(set(band_features_quasi[3][:, feature_id]))) > 1:
            stat, pval = kruskal(band_features_quasi[0][:, feature_id], band_features_quasi[1][:, feature_id],
                                 band_features_quasi[2][:, feature_id], band_features_quasi[3][:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background_by_file_quasi['pval_quasi'].append(pval)
                kruskal_pval_all_background_by_file_quasi['feature_quasi'].append(band_features_names_quasi[feature_id])

        if len(set(band_features_im1[0][:, feature_id]).union(set(band_features_im1[1][:, feature_id])).union(
                set(band_features_im1[2][:, feature_id])).union(set(band_features_im1[3][:, feature_id]))) > 1:
            stat, pval = kruskal(band_features_im1[0][:, feature_id], band_features_im1[1][:, feature_id],
                                 band_features_im1[2][:, feature_id], band_features_im1[3][:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background_by_file_im1['pval_im1'].append(pval)
                kruskal_pval_all_background_by_file_im1['feature_im1'].append(band_features_names_im1[feature_id])

        if len(set(band_features_im2[0][:, feature_id]).union(set(band_features_im2[1][:, feature_id])).union(
                set(band_features_im2[2][:, feature_id])).union(set(band_features_im2[3][:, feature_id]))) > 1:
            stat, pval = kruskal(band_features_im2[0][:, feature_id], band_features_im2[1][:, feature_id],
                                 band_features_im2[2][:, feature_id], band_features_im2[3][:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background_by_file_im2['pval_im2'].append(pval)
                kruskal_pval_all_background_by_file_im2['feature_im2'].append(band_features_names_im2[feature_id])

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background_by_file_real['pval_real'],
                                                             method='fdr_bh')
    kruskal_pval_all_background_by_file_real['pval_bh_real'] = pval_bh

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background_by_file_quasi['pval_quasi'],
                                                             method='fdr_bh')
    kruskal_pval_all_background_by_file_quasi['pval_bh_quasi'] = pval_bh

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background_by_file_im1['pval_im1'],
                                                             method='fdr_bh')
    kruskal_pval_all_background_by_file_im1['pval_bh_im1'] = pval_bh

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background_by_file_im2['pval_im2'],
                                                             method='fdr_bh')
    kruskal_pval_all_background_by_file_im2['pval_bh_im2'] = pval_bh

    df = pd.DataFrame.from_dict(kruskal_pval_all_background_by_file_real)
    df.to_excel("files/kruskal_all_background_by_file_real.xlsx", header=True, index=False)

    df = pd.DataFrame.from_dict(kruskal_pval_all_background_by_file_quasi)
    df.to_excel("files/kruskal_all_background_by_file_quasi.xlsx", header=True, index=False)

    df = pd.DataFrame.from_dict(kruskal_pval_all_background_by_file_im1)
    df.to_excel("files/kruskal_all_background_by_file_im1.xlsx", header=True, index=False)

    df = pd.DataFrame.from_dict(kruskal_pval_all_background_by_file_im2)
    df.to_excel("files/kruskal_all_background_by_file_im2.xlsx", header=True, index=False)

if calculate_all_background_by_class_merged:
    mat_data = []
    num_trials_files = []
    min_trial_len_files = []

    for file_id in range(0, len(new_data_files)):

        file_name = new_data_files[file_id]
        mat_data.append(read_mat(data_path + file_name))

        if file_name.startswith('1st'):
            data_info = mne.create_info(ch_names=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['label'],
                                        sfreq=mat_data[file_id]['subs_ica_bgr'][0]['right_real']['fsample'],
                                        ch_types='eeg')

        num_trials_real, min_trial_len_real = calculate_num_trials(mat_data[file_id], 'right_real')
        num_trials_quasi, min_trial_len_quasi = calculate_num_trials(mat_data[file_id], 'right_quasi')
        num_trials_im1, min_trial_len_im1 = calculate_num_trials(mat_data[file_id], 'right_im1')
        num_trials_im2, min_trial_len_im2 = calculate_num_trials(mat_data[file_id], 'right_im2')

        num_trials = min(num_trials_real, num_trials_quasi, num_trials_im1, num_trials_im2)
        min_trial_len = min(min_trial_len_real, min_trial_len_quasi, min_trial_len_im1, min_trial_len_im2)

        num_trials_files.append(num_trials * 4)
        min_trial_len_files.append(min_trial_len)

    min_trial_len = min(min_trial_len_files)
    num_electrodes = mat_data[0]['subs_ica_bgr'][0]['right_real']['trial'].shape[0]

    data_real = []
    data_quasi = []
    data_im1 = []
    data_im2 = []

    for file_id in range(0, len(new_data_files)):
        data_real.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_quasi.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_im1.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))
        data_im2.append(np.empty(shape=(num_trials_files[file_id], num_electrodes, min_trial_len)))

    freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]

    band_features_real = []
    band_features_quasi = []
    band_features_im1 = []
    band_features_im2 = []

    for file_id in range(0, len(new_data_files)):
        data_real[file_id] = fill_data(data_real[file_id], mat_data[file_id], 'right_real', min_trial_len)
        data_quasi[file_id] = fill_data(data_quasi[file_id], mat_data[file_id], 'right_quasi', min_trial_len)
        data_im1[file_id] = fill_data(data_im1[file_id], mat_data[file_id], 'right_im1', min_trial_len)
        data_im2[file_id] = fill_data(data_im2[file_id], mat_data[file_id], 'right_im2', min_trial_len)

        curr_band_features_real, band_features_names_real = get_band_features(data_real[file_id], freq_bands)
        curr_band_features_quasi, band_features_names_quasi = get_band_features(data_quasi[file_id], freq_bands)
        curr_band_features_im1, band_features_names_im1 = get_band_features(data_im1[file_id], freq_bands)
        curr_band_features_im2, band_features_names_im2 = get_band_features(data_im2[file_id], freq_bands)

        band_features_real.append(curr_band_features_real)
        band_features_quasi.append(curr_band_features_quasi)
        band_features_im1.append(curr_band_features_im1)
        band_features_im2.append(curr_band_features_im2)

    band_features = []
    for file_id in range(0, len(new_data_files)):
        band_features.append(np.concatenate((band_features_real[file_id], band_features_quasi[file_id],
                                             band_features_im1[file_id], band_features_im2[file_id]), axis=0))

    kruskal_pval_all_background_by_file = {'feature': [], 'pval': [], 'pval_bh': []}

    for feature_id in range(0, len(band_features_names_real)):
        if len(set(band_features[0][:, feature_id]).union(set(band_features[1][:, feature_id])).union(
                set(band_features[2][:, feature_id])).union(set(band_features[3][:, feature_id]))) > 1:
            stat, pval = kruskal(band_features[0][:, feature_id], band_features[1][:, feature_id],
                                 band_features[2][:, feature_id], band_features[3][:, feature_id])
            if not np.isnan(pval):
                kruskal_pval_all_background_by_file['pval'].append(pval)
                kruskal_pval_all_background_by_file['feature'].append(band_features_names_real[feature_id])

    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_pval_all_background_by_file['pval'],
                                                             method='fdr_bh')
    kruskal_pval_all_background_by_file['pval_bh'] = pval_bh

    df = pd.DataFrame.from_dict(kruskal_pval_all_background_by_file)
    df.to_excel("files/kruskal_all_background_by_file.xlsx", header=True, index=False)
