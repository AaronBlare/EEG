import pandas as pd
import numpy as np
from pathlib import Path
import mne
from mne.externals.pymatreader import read_mat

data_path = 'E:/YandexDisk/EEG/Data/'
data_files = ['DAY2_SHAM.mat']

movement_type = 'right_real'

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

num_trials = [0, 0]
min_trial_len = mat_data[0][0][movement_type]['trial'][0].shape[1]
for data_id in range(0, 2):
    for subject_id in range(0, len(mat_data[data_id])):
        if data_id == 0:
            for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                num_trials[data_id] += 1
                curr_trial_len = mat_data[data_id][subject_id][movement_type]['trial'][trial_id].shape[1]
                if curr_trial_len < min_trial_len:
                    min_trial_len = curr_trial_len
        else:
            num_trials[data_id] += 1
            curr_trial_len = mat_data[data_id][subject_id][movement_type]['trial'].shape[1]
            if curr_trial_len < min_trial_len:
                min_trial_len = curr_trial_len

num_electrodes = len(data_info['ch_names'])
action_data_V = np.empty(shape=(num_trials[0], num_electrodes, min_trial_len))
rest_data_V = np.empty(shape=(num_trials[1], num_electrodes, min_trial_len))
for data_id in range(0, 2):
    curr_num_trials = 0
    for subject_id in range(0, len(mat_data[data_id])):
        if data_id == 0:
            for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                action_data_V[curr_num_trials, :, :] = \
                    mat_data[data_id][subject_id][movement_type]['trial'][trial_id][:, :min_trial_len]
                curr_num_trials += 1
        else:
            rest_data_V[curr_num_trials, :, :] = \
                mat_data[data_id][subject_id][movement_type]['trial'][:, :min_trial_len]
            curr_num_trials += 1
del mat_data

action_data_uV = action_data_V * 10**6
rest_data_uV = rest_data_V * 10**6
# freq_bands = [('Alpha', 8, 12), ('Low_Beta', 12, 20), ('High_Beta', 20, 30)]
freqs = [8, 30]

action_filtered = mne.filter.filter_data(action_data_uV, data_info['sfreq'], freqs[0], freqs[1])
rest_filtered = mne.filter.filter_data(rest_data_uV, data_info['sfreq'], freqs[0], freqs[1])

action_data = action_filtered ** 2
rest_data = rest_filtered ** 2

action_data_averaged_over_trials = np.mean(action_data, axis=0)
rest_data_averaged_over_trials = np.mean(rest_data, axis=0)

ERDS = (action_data_averaged_over_trials - rest_data_averaged_over_trials) / rest_data_averaged_over_trials

olo = 0
