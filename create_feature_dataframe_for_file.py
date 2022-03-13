import pandas as pd
import numpy as np
from pathlib import Path
import mne
from mne.externals.pymatreader import read_mat
from mne.time_frequency import psd_welch

data_path = 'E:/YandexDisk/EEG/Data/'
data_files = ['DAY2_SHAM.mat']

dataframe_path = 'E:/YandexDisk/EEG/Dataframes/'

background = True
movement_types = ['right_real', 'right_quasi', 'right_im1', 'right_im2']

mat_data = []
for file_id in range(0, len(data_files)):
    curr_mat_data = read_mat(data_path + data_files[file_id])
    mat_data.append(curr_mat_data)
    if file_id == 0:
        if background:
            if 'subs_ica_bgr' in curr_mat_data:
                data_info = mne.create_info(ch_names=curr_mat_data['subs_ica_bgr'][0]['right_real']['label'],
                                            sfreq=curr_mat_data['subs_ica_bgr'][0]['right_real']['fsample'],
                                            ch_types='eeg')
            else:
                data_info = mne.create_info(ch_names=curr_mat_data['res_bgr'][0]['right_real']['label'],
                                            sfreq=curr_mat_data['res_bgr'][0]['right_real']['fsample'],
                                            ch_types='eeg')
        else:
            if 'subs_ica' in curr_mat_data:
                data_info = mne.create_info(ch_names=curr_mat_data['subs_ica'][0]['right_real']['label'],
                                            sfreq=curr_mat_data['subs_ica'][0]['right_real']['fsample'],
                                            ch_types='eeg')
            else:
                data_info = mne.create_info(ch_names=curr_mat_data['res'][0]['right_real']['label'],
                                            sfreq=curr_mat_data['res'][0]['right_real']['fsample'],
                                            ch_types='eeg')
    del curr_mat_data


def calculate_num_trials(data, move_type):
    num_trials = 0
    if background:
        if 'subs_ica_bgr' in data:
            file_flag = 'subs_ica_bgr'
        else:
            file_flag = 'res_bgr'
        min_trial_len = data[file_flag][0][move_type]['trial'].shape[1]
    else:
        if 'subs_ica' in data:
            file_flag = 'subs_ica'
        else:
            file_flag = 'res'
        min_trial_len = data[file_flag][0][move_type]['trial'][0].shape[1]
    for subject_id in range(0, len(data[file_flag])):
        if len(data[file_flag][subject_id]) > 0:
            if background:
                num_trials += 1
            else:
                num_trials += len(data[file_flag][subject_id][move_type]['trial'])
            for trial_id in range(0, len(data[file_flag][subject_id][move_type]['trial'])):
                if background:
                    if data[file_flag][subject_id][move_type]['trial'].shape[1] < min_trial_len:
                        min_trial_len = data[file_flag][subject_id][move_type]['trial'].shape[1]
                else:
                    if data[file_flag][subject_id][move_type]['trial'][trial_id].shape[1] < min_trial_len:
                        min_trial_len = data[file_flag][subject_id][move_type]['trial'][trial_id].shape[1]
    return num_trials, min_trial_len


num_trials_by_movement = {movement_type: 0 for movement_type in movement_types}
min_trial_len_by_movement = []
num_electrodes_by_movement = []

for movement_type in movement_types:
    movement_num_trials = 0
    for file_id in range(0, len(data_files)):
        curr_data = mat_data[file_id]
        curr_num_trials, curr_min_trial_len = calculate_num_trials(curr_data, movement_type)
        if background:
            if 'subs_ica_bgr' in curr_data:
                file_flag = 'subs_ica_bgr'
            else:
                file_flag = 'res_bgr'
            curr_num_electrodes = curr_data[file_flag][0][movement_type]['trial'].shape[0]
        else:
            if 'subs_ica' in curr_data:
                file_flag = 'subs_ica'
            else:
                file_flag = 'res'
            curr_num_electrodes = curr_data[file_flag][0][movement_type]['trial'][0].shape[0]
        min_trial_len_by_movement.append(curr_min_trial_len)
        num_electrodes_by_movement.append(curr_num_electrodes)
        movement_num_trials += curr_num_trials
        del curr_data
    num_trials_by_movement[movement_type] = movement_num_trials

min_trial_len = min(min_trial_len_by_movement)
num_electrodes = min(num_electrodes_by_movement)
trial_start = min_trial_len // 2

data = {
    movement_type: np.empty(shape=(num_trials_by_movement[movement_type], num_electrodes, min_trial_len - trial_start))
    for movement_type in movement_types}


def fill_data(data, raw_data, movement_types, trial_start, trial_len):
    subjects = []
    for movement_type in movement_types:
        curr_trial = 0
        curr_subject = 0
        for file_id in range(0, len(data_files)):
            curr_file = raw_data[file_id]
            if background:
                if 'subs_ica_bgr' in curr_file:
                    file_flag = 'subs_ica_bgr'
                else:
                    file_flag = 'res_bgr'
                for subject_id in range(0, len(curr_file[file_flag])):
                    if len(curr_file[file_flag][subject_id]) > 0:
                        subjects.append(f'S{curr_subject}_T{0}_{movement_type}')
                        data[movement_type][curr_trial, :, :] = \
                            curr_file[file_flag][subject_id][movement_type]['trial'][:, trial_start:trial_len]
                        curr_trial += 1
                    curr_subject += 1
            else:
                if 'subs_ica' in curr_file:
                    file_flag = 'subs_ica'
                else:
                    file_flag = 'res'
                for subject_id in range(0, len(curr_file[file_flag])):
                    if len(curr_file[file_flag][subject_id]) > 0:
                        for trial_id in range(0, len(curr_file[file_flag][subject_id][movement_type]['trial'])):
                            subjects.append(f'S{curr_subject}_T{trial_id}_{movement_type}')
                            data[movement_type][curr_trial, :, :] = \
                                curr_file[file_flag][subject_id][movement_type]['trial'][trial_id][:,
                                trial_start:trial_len]
                            curr_trial += 1
                        curr_subject += 1
            del curr_file
    return data, subjects


data_V, subjects = fill_data(data, mat_data, movement_types, trial_start, min_trial_len)
data = {key: data_V[key] * 10**6 for key in data_V}

freq_bands = [('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]


def get_band_features(data, movement_types, bands):
    num_features_types = 6
    band_features = {movement_type: np.empty(
        shape=(data[movement_type].shape[0], data[movement_type].shape[1] * len(bands) * num_features_types))
        for movement_type in movement_types}
    band_features_names = list()
    for movement_type_id in range(0, len(movement_types)):
        movement_type = movement_types[movement_type_id]
        filtered_epochs = mne.EpochsArray(data=data[movement_type].copy(), info=data_info)
        filtered_epochs.filter(1, 50, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
        psd, freqs = psd_welch(filtered_epochs, fmin=1, fmax=45, tmin=0, tmax=data[movement_types[0]].shape[2])
        psd_band_freqs_ids = [[]]
        curr_band_id = 0
        for freq_id in range(0, freqs.shape[0]):
            freq = freqs[freq_id]
            if freq > bands[curr_band_id][2]:
                psd_band_freqs_ids.append([])
                curr_band_id += 1
            if bands[curr_band_id][1] <= freq <= bands[curr_band_id][2]:
                psd_band_freqs_ids[curr_band_id].append(freq_id)
        feature_id = 0
        for band_id in range(0, len(bands)):
            band = bands[band_id][0]
            f_min = bands[band_id][1]
            f_max = bands[band_id][2]
            filtered_epochs.filter(f_min, f_max, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
            filtered_data = filtered_epochs.get_data()
            for electrode_id in range(0, filtered_data.shape[1]):
                if movement_type_id == 0:
                    electrode_name = filtered_epochs.ch_names[electrode_id]
                    band_features_names.append('_'.join([band, 'mean', electrode_name]))
                    band_features_names.append('_'.join([band, 'median', electrode_name]))
                    band_features_names.append('_'.join([band, 'std', electrode_name]))
                    band_features_names.append('_'.join([band, 'max', electrode_name]))
                    band_features_names.append('_'.join([band, 'min', electrode_name]))
                    band_features_names.append('_'.join([band, 'psd', electrode_name]))
                for trial_id in range(0, filtered_data.shape[0]):
                    band_features[movement_type][trial_id, feature_id] = np.mean(
                        filtered_data[trial_id, electrode_id, :])
                    band_features[movement_type][trial_id, feature_id + 1] = np.median(
                        filtered_data[trial_id, electrode_id, :])
                    band_features[movement_type][trial_id, feature_id + 2] = np.std(
                        filtered_data[trial_id, electrode_id, :])
                    band_features[movement_type][trial_id, feature_id + 3] = np.max(
                        filtered_data[trial_id, electrode_id, :])
                    band_features[movement_type][trial_id, feature_id + 4] = np.min(
                        filtered_data[trial_id, electrode_id, :])
                    band_features[movement_type][trial_id, feature_id + 5] = np.mean(
                        psd[trial_id, electrode_id, psd_band_freqs_ids[band_id]])
                feature_id += num_features_types
        del filtered_epochs
    return band_features, band_features_names


band_features, band_features_names = get_band_features(data, movement_types, freq_bands)

features = np.concatenate([band_features[movement_type] for movement_type in movement_types], axis=0)
subjects_features = np.concatenate((np.c_[subjects], features), axis=1)

features_names = band_features_names

classes = []
classes_names = []
for movement_type_id in range(0, len(movement_types)):
    movement_type = movement_types[movement_type_id]
    classes.extend([movement_type_id] * band_features[movement_type].shape[0])
    classes_names.extend([movement_type] * band_features[movement_type].shape[0])

df = pd.DataFrame(np.concatenate((subjects_features, np.c_[classes_names]), axis=1),
                  columns=['trial'] + features_names + ['class'])

Path(dataframe_path).mkdir(parents=True, exist_ok=True)
data_files_wo_extension = [data_file[:-4] for data_file in data_files]
if background:
    df.to_excel(f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}_background_uV.xlsx",
                header=True, index=False)
else:
    df.to_excel(f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}_uV.xlsx", header=True, index=False)
