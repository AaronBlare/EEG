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

num_epochs = 0
min_epoch_len = mat_data['res'][0]['right_real']['trial'][0].shape[1]
for subject_id in range(0, len(mat_data['res'])):
    if len(mat_data['res'][subject_id]) > 0:
        num_epochs += len(mat_data['res'][0]['right_real']['trial'])
        for epoch_id in range(0, len(mat_data['res'][0]['right_real']['trial'])):
            if mat_data['res'][subject_id]['right_real']['trial'][epoch_id].shape[1] < min_epoch_len:
                min_epoch_len = mat_data['res'][subject_id]['right_real']['trial'][epoch_id].shape[1]

real_data = np.empty(shape=(num_epochs, mat_data['res'][0]['right_real']['trial'][0].shape[0], min_epoch_len))

curr_epoch = 0
for subject_id in range(0, len(mat_data['res'])):
    if len(mat_data['res'][subject_id]) > 0:
        for epoch_id in range(0, len(mat_data['res'][0]['right_real']['trial'])):
            real_data[curr_epoch, :, :] = mat_data['res'][subject_id]['right_real']['trial'][epoch_id][:,
                                          - min_epoch_len:]
            curr_epoch += 1

real_epochs = mne.EpochsArray(data=real_data, info=data_info)

freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]
frequency_map = list()
for band, f_min, f_max in freq_bands:
    '''
    mat_data = read_mat(data_path + data_file)
    real_data = np.empty(shape=(num_epochs, mat_data['res'][0]['right_real']['trial'][0].shape[0], min_epoch_len))

    curr_epoch = 0
    for subject_id in range(0, len(mat_data['res'])):
        if len(mat_data['res'][subject_id]) > 0:
            for epoch_id in range(0, len(mat_data['res'][0]['right_real']['trial'])):
                real_data[curr_epoch, :, :] = mat_data['res'][subject_id]['right_real']['trial'][epoch_id][:,
                                              - min_epoch_len:]
                curr_epoch += 1
    '''
    filtered_epochs = mne.EpochsArray(data=real_data.copy(), info=data_info)
    filtered_epochs.filter(f_min, f_max, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
    filtered_epochs.subtract_evoked()
    filtered_epochs.apply_hilbert(envelope=True)
    frequency_map.append(((band, f_min, f_max), filtered_epochs.average()))
    del filtered_epochs

'''
def stat_fun(x):
    """Return sum of squares."""
    return np.sum(x ** 2, axis=0)


# Plot
fig, axes = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
colors = plt.get_cmap('winter_r')(np.linspace(0, 1, 5))
for ((freq_name, fmin, fmax), average), color, ax in zip(
        frequency_map, colors, axes.ravel()[::-1]):
    times = average.times * 1e3
    gfp = np.sum(average.data ** 2, axis=0)
    gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
    ax.plot(times, gfp, label=freq_name, color=color, linewidth=2.5)
    ax.axhline(0, linestyle='--', color='grey', linewidth=2)
    ci_low, ci_up = bootstrap_confidence_interval(average.data, random_state=0,
                                                  stat_fun=stat_fun)
    ci_low = rescale(ci_low, average.times, baseline=(None, 0))
    ci_up = rescale(ci_up, average.times, baseline=(None, 0))
    ax.fill_between(times, gfp + ci_up, gfp - ci_low, color=color, alpha=0.3)
    ax.grid(True)
    ax.set_ylabel('GFP')
    ax.annotate('%s (%d-%dHz)' % (freq_name, fmin, fmax),
                xy=(0.95, 0.8),
                horizontalalignment='right',
                xycoords='axes fraction')

axes.ravel()[-1].set_xlabel('Time [ms]')
plt.show()
'''
ssd = mne.decoding.SSD(info=data_info,
                       reg='oas',
                       filt_params_signal=dict(l_freq=35, h_freq=40,
                                               l_trans_bandwidth=1, h_trans_bandwidth=1),
                       filt_params_noise=dict(l_freq=34, h_freq=41,
                                              l_trans_bandwidth=1, h_trans_bandwidth=1))
ssd.fit(X=real_data)
ssd_sources = ssd.transform(X=real_data)

psds, freqs = mne.time_frequency.psd_array_welch(ssd_sources, sfreq=data_info['sfreq'])
spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)
'''
fig, ax = plt.subplots(1)
ax.plot(spec_ratio, color='black')
ax.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
ax.set_xlabel("Eigenvalue Index")
ax.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
ax.legend()
ax.axhline(1, linestyle='--')
plt.show()
'''

pca = mne.decoding.UnsupervisedSpatialFilter(PCA(30), average=False)
pca_data = pca.fit_transform(real_data)
ev = mne.EvokedArray(np.mean(pca_data, axis=0),
                     mne.create_info(30, data_info['sfreq'], ch_types='eeg'))
ev.plot(window_title="PCA", time_unit='s')

ica = mne.decoding.UnsupervisedSpatialFilter(FastICA(30), average=False)
ica_data = ica.fit_transform(real_data)
ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      mne.create_info(30, data_info['sfreq'], ch_types='eeg'))
ev1.plot(show=False, window_title='ICA', time_unit='s')

plt.show()
