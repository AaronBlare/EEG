import pandas as pd
import numpy as np
from pathlib import Path
import mne
from mne.externals.pymatreader import read_mat
import plotly.graph_objects as go

data_path = 'E:/YandexDisk/EEG/Data/'
data_files = ['DAY2_SHAM.mat']

figures_path = 'E:/YandexDisk/EEG/Figures/ERDS/'
suffix = '2nd_Day_sham'

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

num_trials_by_movement = {move_type: 0 for move_type in movement_types}
min_trial_len = mat_data[0][0][movement_types[0]]['trial'][0].shape[1]
num_trials_by_subject = {move_type: [] for move_type in movement_types}
for data_id in range(0, len(mat_data)):
    for subject_id in range(0, len(mat_data[data_id])):
        for movement_type in movement_types:
            num_trials_by_subject[movement_type].append(0)
            for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                num_trials_by_movement[movement_type] += 1
                num_trials_by_subject[movement_type][subject_id] += 1
                curr_trial_len = mat_data[data_id][subject_id][movement_type]['trial'][trial_id].shape[1]
                if curr_trial_len < min_trial_len:
                    min_trial_len = curr_trial_len

num_electrodes = len(data_info['ch_names'])
action_start = min_trial_len // 2
action_trial_len = min_trial_len - action_start
rest_trial_len = action_start + 1
action_data_V = {move_type: np.empty(shape=(num_trials_by_movement[move_type], num_electrodes, action_trial_len)) for
                 move_type in movement_types}
rest_data_V = {move_type: np.empty(shape=(num_trials_by_movement[move_type], num_electrodes, rest_trial_len)) for
               move_type in movement_types}
for data_id in range(0, len(mat_data)):
    for movement_type in movement_types:
        curr_num_trials = 0
        for subject_id in range(0, len(mat_data[data_id])):
            for trial_id in range(0, len(mat_data[data_id][subject_id][movement_type]['trial'])):
                action_data_V[movement_type][curr_num_trials, :, :] = \
                    mat_data[data_id][subject_id][movement_type]['trial'][trial_id][:, action_start:min_trial_len]
                rest_data_V[movement_type][curr_num_trials, :, :] = \
                    mat_data[data_id][subject_id][movement_type]['trial'][trial_id][:, :rest_trial_len]
                curr_num_trials += 1
del mat_data

action_data_uV = {move_type: action_data_V[move_type] * (10 ** 6) for move_type in movement_types}
rest_data_uV = {move_type: rest_data_V[move_type] * (10 ** 6) for move_type in movement_types}

freqs_1D = [8, 30]

action_data_1D = {
    move_type: mne.filter.filter_data(action_data_uV[move_type], data_info['sfreq'], freqs_1D[0], freqs_1D[1])
    for move_type in movement_types}
rest_data_1D = {
    move_type: mne.filter.filter_data(rest_data_uV[move_type], data_info['sfreq'], freqs_1D[0], freqs_1D[1])
    for move_type in movement_types}

action_data_1D = {move_type: action_data_1D[move_type] ** 2 for move_type in movement_types}
rest_data_1D = {move_type: rest_data_1D[move_type] ** 2 + 10 ** (-16) for move_type in movement_types}

action_data_averaged_over_trials_1D = {move_type: [] for move_type in movement_types}
rest_data_averaged_over_trials_1D = {move_type: [] for move_type in movement_types}
for movement_type in movement_types:
    for subject_id in range(0, len(num_trials_by_subject[movement_type])):
        start_trial = num_trials_by_subject[movement_type][subject_id] * subject_id
        finish_trial = num_trials_by_subject[movement_type][subject_id] * (subject_id + 1)
        action_data_averaged_over_trials_1D[movement_type].append(
            np.mean(action_data_1D[movement_type][start_trial:finish_trial], axis=0))
        rest_data_averaged_over_trials_1D[movement_type].append(
            np.mean(rest_data_1D[movement_type][start_trial:finish_trial], axis=0))

averaging_window = 50
ERDS_subjects_1D = {move_type: [] for move_type in movement_types}
for movement_type in movement_types:
    for subject_id in range(0, len(num_trials_by_subject[movement_type])):
        curr_action = action_data_averaged_over_trials_1D[movement_type][subject_id]
        curr_rest = rest_data_averaged_over_trials_1D[movement_type][subject_id]
        curr_action_smoothed = np.empty(shape=(curr_action.shape[0], curr_action.shape[1] - averaging_window + 1))
        curr_rest_smoothed = np.empty(shape=(curr_rest.shape[0], curr_rest.shape[1] - averaging_window + 1))
        for electrode_id in range(0, num_electrodes):
            curr_action_smoothed[electrode_id, :] = np.convolve(curr_action[electrode_id, :],
                                                                np.ones(averaging_window) / averaging_window,
                                                                mode='valid')
            curr_rest_smoothed[electrode_id, :] = np.convolve(curr_rest[electrode_id, :],
                                                              np.ones(averaging_window) / averaging_window,
                                                              mode='valid')
        ERDS_subjects_1D[movement_type].append((curr_action_smoothed - curr_rest_smoothed) / curr_rest_smoothed)

ERDS_trials_1D = {}
for movement_type in movement_types:
    ERDS_trials_1D[movement_type] = (action_data_1D[movement_type] - rest_data_1D[movement_type]) / rest_data_1D[
        movement_type]

for electrode_id in range(0, num_electrodes):
    for movement_type in movement_types:
        for subject_id in range(0, len(num_trials_by_subject[movement_type])):
            ERDS_len = ERDS_subjects_1D[movement_type][subject_id].shape[1]

            curr_x = np.linspace(0, ERDS_len / 1000, ERDS_len)
            curr_y = ERDS_subjects_1D[movement_type][subject_id][electrode_id, :]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curr_x, y=curr_y, mode='lines'))
            fig.add_annotation(dict(font=dict(color="black", size=14), x=0.5, y=-0.1, showarrow=False,
                                    text="Time, s", xref="paper", yref="paper"))
            fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.33, y=0.5, showarrow=False,
                                    text="ERDS", textangle=-90, xref="paper", yref="paper"))

            curr_title = f"Electrode {data_info['ch_names'][electrode_id]}, Subject {subject_id + 1}"
            fig.update_layout(margin=dict(t=50, l=200), title=curr_title)

            curr_path = f"{figures_path}/{suffix}/{movement_type}/1D/subjects/{subject_id + 1}"
            Path(curr_path).mkdir(parents=True, exist_ok=True)
            fig.write_image(f"{curr_path}/{data_info['ch_names'][electrode_id]}.png")
            fig.write_image(f"{curr_path}/{data_info['ch_names'][electrode_id]}.pdf")

        for trial_id in range(0, num_trials_by_movement[movement_type]):
            curr_x = np.linspace(0, action_trial_len / 1000, action_trial_len)
            curr_y = ERDS_trials_1D[movement_type][trial_id, electrode_id, :]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curr_x, y=curr_y, mode='lines'))

            fig.add_annotation(dict(font=dict(color="black", size=14), x=0.5, y=-0.1, showarrow=False,
                                    text="Time, s", xref="paper", yref="paper"))
            fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.33, y=0.5, showarrow=False,
                                    text="ERDS", textangle=-90, xref="paper", yref="paper"))

            curr_title = f"Electrode {data_info['ch_names'][electrode_id]}, Trial {trial_id + 1}"
            fig.update_layout(margin=dict(t=50, l=200), title=curr_title)

            curr_path = f"{figures_path}/{suffix}/{movement_type}/1D/trials/{trial_id + 1}"
            Path(curr_path).mkdir(parents=True, exist_ok=True)
            fig.write_image(f"{curr_path}/{data_info['ch_names'][electrode_id]}.png")
            fig.write_image(f"{curr_path}/{data_info['ch_names'][electrode_id]}.pdf")

del action_data_V, rest_data_V, action_data_1D, rest_data_1D

freqs_2D = [1, 60]
num_freqs = freqs[1] - freqs[0]

ERDS_2D = {move_type: np.empty(shape=(num_freqs, num_electrodes, min_trial_len)) for move_type in movement_types}

for electrode_id in range(0, num_electrodes):
    for freq_id in range(0, num_freqs):
        freq_start = freqs[0] + freq_id
        freq_finish = freqs[0] + freq_id + 1
        for movement_type in movement_types:
            Path(f"{figures_path}/{suffix}/{movement_type}/2D/").mkdir(parents=True, exist_ok=True)
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
            ERDS_2D[movement_type][freq_id, electrode_id, :] = (
                                                                       curr_electrode_freq_action_averaged - curr_electrode_freq_rest_averaged) / curr_electrode_freq_rest_averaged

    for movement_type in movement_types:
        fig = go.Figure(data=go.Heatmap(x=np.linspace(0, min_trial_len / 1000, min_trial_len),
                                        y=np.linspace(freqs[0], freqs[1], num_freqs),
                                        z=ERDS_2D[movement_type][:, electrode_id, :]))
        fig.add_annotation(dict(font=dict(color="black", size=14), x=0.5, y=-0.1, showarrow=False,
                                text="Time, s", xref="paper", yref="paper"))
        fig.add_annotation(dict(font=dict(color="black", size=14), x=-0.33, y=0.5, showarrow=False,
                                text="Frequency, Hz", textangle=-90, xref="paper", yref="paper"))
        fig.update_layout(margin=dict(t=50, l=200))
        fig.write_image(f"{figures_path}/{suffix}/{movement_type}/2D/{data_info['ch_names'][electrode_id]}.png")
        fig.write_image(f"{figures_path}/{suffix}/{movement_type}/2D/{data_info['ch_names'][electrode_id]}.pdf")

olo = 0
