import pandas as pd
import numpy as np
from pathlib import Path
import mne
from mne.externals.pymatreader import read_mat

data_path = 'E:/YandexDisk/EEG/Data/'
data_files = ['DAY2_SHAM.mat']

movement_types = ['right_real', 'right_quasi', 'right_im1', 'right_im2']

mat_data = []
curr_mat_data = read_mat(data_path + data_files[0])

if 'subs_ica' in curr_mat_data:
    mat_data.append(curr_mat_data['subs_ica'])
    data_info = mne.create_info(ch_names=curr_mat_data['subs_ica'][0]['right_real']['label'],
                                sfreq=curr_mat_data['subs_ica'][0]['right_real']['fsample'],
                                ch_types='eeg')
else:
    mat_data.append(curr_mat_data['res'])
    data_info = mne.create_info(ch_names=curr_mat_data['res'][0]['right_real']['label'],
                                sfreq=curr_mat_data['res'][0]['right_real']['fsample'],
                                ch_types='eeg')

if 'res_bgr' in curr_mat_data:
    mat_data.append(curr_mat_data['res_bgr'])
else:
    curr_mat_data = read_mat(data_path + data_files[1])
    mat_data.append(curr_mat_data['subs_ica_bgr'])
del curr_mat_data

num_trials_by_movement = {move_type: [0, 0] for move_type in movement_types}
min_trial_len = mat_data[0][0][movement_types[0]]['trial'][0].shape[1]
for data_id in range(0, 2):
    for subject_id in range(0, len(mat_data[data_id])):
        for movement_type in movement_types:
            if data_id == 0:
                for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                    num_trials_by_movement[movement_type][data_id] += 1
                    curr_trial_len = mat_data[data_id][subject_id][movement_type]['trial'][trial_id].shape[1]
                    if curr_trial_len < min_trial_len:
                        min_trial_len = curr_trial_len
            else:
                num_trials_by_movement[movement_type][data_id] += 1
                curr_trial_len = mat_data[data_id][subject_id][movement_type]['trial'].shape[1]
                if curr_trial_len < min_trial_len:
                    min_trial_len = curr_trial_len

num_electrodes = len(data_info['ch_names'])
action_data_V = {move_type: np.empty(shape=(num_trials_by_movement[move_type][0], num_electrodes, min_trial_len)) for
                 move_type in movement_types}
rest_data_V = {move_type: np.empty(shape=(num_trials_by_movement[move_type][1], num_electrodes, min_trial_len)) for
               move_type in movement_types}
for data_id in range(0, 2):
    for movement_type in movement_types:
        curr_num_trials = 0
        for subject_id in range(0, len(mat_data[data_id])):
            if data_id == 0:
                for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                    action_data_V[movement_type][curr_num_trials, :, :] = \
                        mat_data[data_id][subject_id][movement_type]['trial'][trial_id][:, :min_trial_len]
                    curr_num_trials += 1
            else:
                rest_data_V[movement_type][curr_num_trials, :, :] = \
                    mat_data[data_id][subject_id][movement_type]['trial'][:, :min_trial_len]
                curr_num_trials += 1
del mat_data

action_data_uV = {move_type: action_data_V[move_type] * (10 ** 6) for move_type in movement_types}
rest_data_uV = {move_type: rest_data_V[move_type] * (10 ** 6) for move_type in movement_types}
# freq_bands = [('Alpha', 8, 12), ('Low_Beta', 12, 20), ('High_Beta', 20, 30)]
freqs = [8, 30]
num_freqs = freqs[1] - freqs[0]

action_data_1D = {
    move_type: mne.filter.filter_data(action_data_uV[move_type], data_info['sfreq'], freqs[0], freqs[1])
    for move_type in movement_types}
rest_data_1D = {move_type: mne.filter.filter_data(rest_data_uV[move_type], data_info['sfreq'], freqs[0], freqs[1])
                for move_type in movement_types}

action_data_1D = {move_type: action_data_1D[move_type] ** 2 for move_type in movement_types}
rest_data_1D = {move_type: rest_data_1D[move_type] ** 2 for move_type in movement_types}

action_data_averaged_over_trials_1D = {move_type: np.mean(action_data_1D[move_type], axis=0)
                                       for move_type in movement_types}
rest_data_averaged_over_trials_1D = {move_type: np.mean(rest_data_1D[move_type], axis=0)
                                     for move_type in movement_types}

ERDS_1D = {move_type: (action_data_averaged_over_trials_1D[move_type] - rest_data_averaged_over_trials_1D[move_type]) /
                      rest_data_averaged_over_trials_1D[move_type] for move_type in movement_types}

del action_data_V, rest_data_V, action_data_1D, rest_data_1D

ERDS_2D = {move_type: np.empty(shape=(num_freqs, num_electrodes, min_trial_len)) for move_type in movement_types}

for electrode_id in range(0, num_electrodes):
    for freq_id in range(0, num_freqs):
        freq_start = freqs[0] + freq_id
        freq_finish = freqs[0] + freq_id + 1
        for movement_type in movement_types:
            curr_electrode_freq_action = mne.filter.filter_data(action_data_uV[movement_type][:, electrode_id, :],
                                                                data_info['sfreq'],
                                                                freq_start, freq_finish)
            curr_electrode_freq_rest = mne.filter.filter_data(rest_data_uV[movement_type][:, electrode_id, :],
                                                              data_info['sfreq'],
                                                              freq_start, freq_finish)

            curr_electrode_freq_action **= 2
            curr_electrode_freq_rest **= 2

            curr_electrode_freq_action_averaged = np.mean(curr_electrode_freq_action, axis=0)
            curr_electrode_freq_rest_averaged = np.mean(curr_electrode_freq_rest, axis=0)
            ERDS_2D[movement_type][freq_id, electrode_id, :] = (curr_electrode_freq_action_averaged - curr_electrode_freq_rest_averaged) / curr_electrode_freq_rest_averaged

olo = 0
