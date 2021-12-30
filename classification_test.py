import numpy as np
import mne
from mne.externals.pymatreader import read_mat
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import pandas as pd
from catboost import CatBoost
import plotly.figure_factory as ff
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from pathlib import Path

shap.initjs()

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

data_right_real = np.empty(shape=(num_epochs, min_time, min_epoch_len - 5000))
data_right_quasi = np.empty(shape=(num_epochs, min_time, min_epoch_len - 5000))
data_right_im1 = np.empty(shape=(num_epochs, min_time, min_epoch_len - 5000))
data_right_im2 = np.empty(shape=(num_epochs, min_time, min_epoch_len - 5000))


def fill_data(data, raw_data, move_type, epoch_len):
    curr_epoch = 0
    for subject_id in range(0, len(raw_data['res'])):
        if len(raw_data['res'][subject_id]) > 0:
            for epoch_id in range(0, len(raw_data['res'][0][move_type]['trial'])):
                data[curr_epoch, :, :] = raw_data['res'][subject_id][move_type]['trial'][epoch_id][:, 5000:epoch_len]
                curr_epoch += 1
    return data


data_right_real = fill_data(data_right_real, mat_data, 'right_real', min_epoch_len)
data_right_quasi = fill_data(data_right_quasi, mat_data, 'right_quasi', min_epoch_len)
data_right_im1 = fill_data(data_right_im1, mat_data, 'right_im1', min_epoch_len)
data_right_im2 = fill_data(data_right_im2, mat_data, 'right_im2', min_epoch_len)

all_data = np.concatenate((data_right_real, data_right_quasi, data_right_im1, data_right_im2), axis=0)

freq_bands = [('Delta', 0, 4), ('Theta', 4, 8), ('Alpha', 8, 12), ('Beta', 12, 30), ('Gamma', 30, 45)]


def get_band_features(data, bands):
    band_features = np.empty(shape=(data.shape[0], data.shape[1] * 5 * 5))
    band_features_names = list()
    band_id = 0
    for band, f_min, f_max in bands:
        filtered_epochs = mne.EpochsArray(data=data.copy(), info=data_info)
        filtered_epochs.filter(f_min, f_max, n_jobs=1, l_trans_bandwidth=1, h_trans_bandwidth=1)
        filtered_epochs.subtract_evoked()
        filtered_epochs.apply_hilbert(envelope=True)
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
'''

all_features = np.concatenate((band_features_right_real, band_features_right_quasi,
                               band_features_right_im1, band_features_right_im2), axis=0)
all_names = band_features_names_right_real
all_classes = [0] * band_features_right_real.shape[0] + [1] * band_features_right_quasi.shape[0] + \
              [2] * band_features_right_im1.shape[0] + [3] * band_features_right_im2.shape[0]

all_classes_names = ['real'] * band_features_right_real.shape[0] + ['quasi'] * band_features_right_quasi.shape[0] + \
                    ['im1'] * band_features_right_im1.shape[0] + ['im2'] * band_features_right_im2.shape[0]

df = pd.DataFrame(np.concatenate((all_features, np.c_[all_classes_names]), axis=1),
                  columns=all_names + ['class'])
df.to_excel("dataframe.xlsx", header=True, index=False)

ids_train, ids_val = train_test_split(np.arange(len(all_classes)),
                                      test_size=0.2,
                                      stratify=all_classes)

train_features = all_features[ids_train, :]
val_features = all_features[ids_val, :]

train_classes = [all_classes[i] for i in list(ids_train)]
val_classes = [all_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi', 'im1', 'im2']

model_params = {'classes_count': 4,
                'loss_function': 'MultiClass',
                'learning_rate': 0.03,
                'depth': 6,
                'min_data_in_leaf': 1,
                'max_leaves': 31,
                'verbose': 1,
                'iterations': 1000,
                'early_stopping_rounds': 100}

model = CatBoost(params=model_params)
model.fit(train_features, train_classes, eval_set=(val_features, val_classes))
model.set_feature_names(all_names)
# model.save_model(f"epoch_{model.best_iteration_}.model")

train_pred = model.predict(train_features, prediction_type="Class")
val_pred = model.predict(val_features, prediction_type="Class")
val_pred_probs = model.predict(val_features, prediction_type="Probability")

y_train_real = train_classes
y_train_pred = [item[0] for item in train_pred]
y_val_real = val_classes
y_val_pred = [item[0] for item in val_pred]

is_correct_pred = (np.array(y_val_real) == np.array(y_val_pred))
mistakes_ids = np.where(is_correct_pred == False)[0]

metrics_dict = {'train': [], 'val': []}

m_val = f1_score(y_train_real, y_train_pred, average='weighted')
metrics_dict['train'].append(m_val)
m_val = f1_score(y_val_real, y_val_pred, average='weighted')
metrics_dict['val'].append(m_val)
'''
m_val = roc_auc_score(y_train_real, y_train_pred, average='weighted', multi_class='ovo')
metrics_dict['train'].append(m_val)
m_val = roc_auc_score(y_val_real, y_val_pred, average='weighted', multi_class='ovo')
metrics_dict['val'].append(m_val)
'''
m_val = accuracy_score(y_train_real, y_train_pred)
metrics_dict['train'].append(m_val)
m_val = accuracy_score(y_val_real, y_val_pred)
metrics_dict['val'].append(m_val)


def proba(X):
    y = model.predict(X, prediction_type='Probability')
    return y


explainer = shap.KernelExplainer(proba, data=val_features)
shap_values = explainer.shap_values(val_features)

shap.summary_plot(
    shap_values=shap_values,
    features=val_features,
    feature_names=all_names,
    max_display=30,
    class_names=classes_names,
    class_inds=list(range(len(classes_names))),
    plot_size=(18, 10),
    show=False,
    color=plt.get_cmap("Set1")
)
plt.savefig('figures/SHAP_bar.png')
plt.savefig('figures/SHAP_bar.pdf')
plt.close()

for st_id, st in enumerate(classes_names):
    shap.summary_plot(
        shap_values=shap_values[st_id],
        features=val_features,
        feature_names=all_names,
        max_display=30,
        plot_size=(18, 10),
        plot_type="violin",
        title=st,
        show=False
    )
    plt.savefig(f"figures/{st}_beeswarm.png")
    plt.savefig(f"figures/{st}_beeswarm.pdf")
    plt.close()

for m_id in mistakes_ids:
    subj_cl = y_val_real[m_id]
    subj_pred_cl = y_val_pred[m_id]
    for st_id, st in enumerate(classes_names):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[st_id][m_id],
                base_values=explainer.expected_value[st_id],
                data=val_features[m_id],
                feature_names=all_names
            ),
            max_display=30,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(18, 10, forward=True)
        Path(f"figures/errors/real({classes_names[subj_cl]})_pred({classes_names[subj_pred_cl]})/{m_id}").mkdir(parents=True,
                                                                                                        exist_ok=True)
        fig.savefig(
            f"figures/errors/real({classes_names[subj_cl]})_pred({classes_names[subj_pred_cl]})/{m_id}/waterfall_{st}.pdf")
        fig.savefig(
            f"figures/errors/real({classes_names[subj_cl]})_pred({classes_names[subj_pred_cl]})/{m_id}/waterfall_{st}.png")
        plt.close()

passed_examples = {x: 0 for x in range(len(classes_names))}
for subj_id in range(val_features.shape[0]):
    subj_cl = y_val_real[subj_id]
    if passed_examples[subj_cl] < len(y_train_real):
        for st_id, st in enumerate(classes_names):
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][subj_id],
                    base_values=explainer.expected_value[st_id],
                    data=val_features[subj_id],
                    feature_names=all_names
                ),
                max_display=30,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 10, forward=True)
            Path(f"figures/{classes_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}").mkdir(parents=True,
                                                                                                 exist_ok=True)
            fig.savefig(f"figures/{classes_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.pdf")
            fig.savefig(f"figures/{classes_names[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.png")
            plt.close()
        passed_examples[subj_cl] += 1

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)


def save_figure(fig, fn, width=800, height=600, scale=2):
    fig.write_image(f"figures/{fn}.png")
    fig.write_image(f"figures/{fn}.pdf")


fig = ff.create_annotated_heatmap(conf_mtx_val, x=classes_names, y=classes_names, colorscale='Viridis')
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=-0.33,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
fig.update_layout(margin=dict(t=50, l=200))
fig['data'][0]['showscale'] = True
save_figure(fig, "confusion_matrix_val_cat")

metrics_df = pd.DataFrame.from_dict(metrics_dict)
metrics_df.to_excel("figures/metrics_cat.xlsx", index=True)

########################################

binary_features = np.concatenate((band_features_right_real, band_features_right_quasi), axis=0)
binary_names = band_features_names_right_real
binary_classes = [0] * band_features_right_real.shape[0] + [1] * band_features_right_quasi.shape[0]

ids_train, ids_val = train_test_split(np.arange(len(binary_classes)),
                                      test_size=0.2,
                                      stratify=binary_classes)

train_features = binary_features[ids_train, :]
val_features = binary_features[ids_val, :]

train_classes = [binary_classes[i] for i in list(ids_train)]
val_classes = [binary_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi']

model_params = {'loss_function': 'Logloss',
                'learning_rate': 0.01,
                'depth': 6,
                'min_data_in_leaf': 1,
                'max_leaves': 31,
                'verbose': 1,
                'iterations': 1000,
                'early_stopping_rounds': 100}

model = CatBoost(params=model_params)
model.fit(train_features, train_classes, eval_set=(val_features, val_classes))
model.set_feature_names(all_names)
# model.save_model(f"epoch_{model.best_iteration_}.model")

train_pred = model.predict(train_features, prediction_type="Class")
val_pred = model.predict(val_features, prediction_type="Class")

y_train_real = train_classes
y_train_pred = train_pred
y_val_real = val_classes
y_val_pred = val_pred

metrics_dict = {'train': [], 'val': []}

m_val = f1_score(y_train_real, y_train_pred, average='weighted')
metrics_dict['train'].append(m_val)
m_val = f1_score(y_val_real, y_val_pred, average='weighted')
metrics_dict['val'].append(m_val)

m_val = accuracy_score(y_train_real, y_train_pred)
metrics_dict['train'].append(m_val)
m_val = accuracy_score(y_val_real, y_val_pred)
metrics_dict['val'].append(m_val)

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)

fig = ff.create_annotated_heatmap(conf_mtx_val, x=classes_names, y=classes_names, colorscale='Viridis')
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=-0.33,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
fig.update_layout(margin=dict(t=50, l=200))
fig['data'][0]['showscale'] = True
save_figure(fig, "confusion_matrix_val_cat_binary")

metrics_df = pd.DataFrame.from_dict(metrics_dict)
metrics_df.to_excel("figures/metrics_cat_binary.xlsx", index=True)

########################################

ids_train, ids_val = train_test_split(np.arange(len(all_classes)),
                                      test_size=0.2,
                                      stratify=all_classes)

train_features = all_features[ids_train, :]
val_features = all_features[ids_val, :]

train_classes = [all_classes[i] for i in list(ids_train)]
val_classes = [all_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi', 'im1', 'im2']

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=all_names)
dmat_val = xgb.DMatrix(val_features, val_classes, feature_names=all_names)

model_params = {
    'num_class': 4,
    'booster': 'gbtree',
    'eta': 0.3,
    'max_depth': 6,
    'gamma': 0,
    'sampling_method': 'uniform',
    'subsample': 1,
    'objective': 'multi:softprob',
    'verbosity': 1,
}

num_boost_round = 1000
early_stopping_rounds = 100
bst = xgb.train(
    params=model_params,
    dtrain=dmat_train,
    evals=[(dmat_train, "train"), (dmat_val, "val")],
    num_boost_round=num_boost_round,
    early_stopping_rounds=early_stopping_rounds
)
# bst.save_model(f"epoch_{bst.best_iteration}.model")

train_pred = bst.predict(dmat_train)
val_pred = bst.predict(dmat_val)

y_train_real = train_classes
y_train_pred = np.argmax(train_pred, 1)
y_val_real = val_classes
y_val_pred = np.argmax(val_pred, 1)

metrics_dict = {'train': [], 'val': []}

m_val = f1_score(y_train_real, y_train_pred, average='weighted')
metrics_dict['train'].append(m_val)
m_val = f1_score(y_val_real, y_val_pred, average='weighted')
metrics_dict['val'].append(m_val)

m_val = accuracy_score(y_train_real, y_train_pred)
metrics_dict['train'].append(m_val)
m_val = accuracy_score(y_val_real, y_val_pred)
metrics_dict['val'].append(m_val)

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)

fig = ff.create_annotated_heatmap(conf_mtx_val, x=classes_names, y=classes_names, colorscale='Viridis')
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=-0.33,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
fig.update_layout(margin=dict(t=50, l=200))
fig['data'][0]['showscale'] = True
save_figure(fig, "confusion_matrix_val_xgb")

metrics_df = pd.DataFrame.from_dict(metrics_dict)
metrics_df.to_excel("figures/metrics_xgb.xlsx", index=True)

########################################

ids_train, ids_val = train_test_split(np.arange(len(binary_classes)),
                                      test_size=0.2,
                                      stratify=binary_classes)

train_features = binary_features[ids_train, :]
val_features = binary_features[ids_val, :]

train_classes = [binary_classes[i] for i in list(ids_train)]
val_classes = [binary_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi']

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=all_names)
dmat_val = xgb.DMatrix(val_features, val_classes, feature_names=all_names)

model_params = {
    'num_class': 2,
    'booster': 'gbtree',
    'eta': 0.3,
    'max_depth': 6,
    'gamma': 0,
    'sampling_method': 'uniform',
    'subsample': 1,
    'objective': 'multi:softprob',
    'verbosity': 1,
}

num_boost_round = 1000
early_stopping_rounds = 100
bst = xgb.train(
    params=model_params,
    dtrain=dmat_train,
    evals=[(dmat_train, "train"), (dmat_val, "val")],
    num_boost_round=num_boost_round,
    early_stopping_rounds=early_stopping_rounds
)
# bst.save_model(f"epoch_{bst.best_iteration}.model")

train_pred = bst.predict(dmat_train)
val_pred = bst.predict(dmat_val)

y_train_real = train_classes
y_train_pred = np.argmax(train_pred, 1)
y_val_real = val_classes
y_val_pred = np.argmax(val_pred, 1)

metrics_dict = {'train': [], 'val': []}

m_val = f1_score(y_train_real, y_train_pred, average='weighted')
metrics_dict['train'].append(m_val)
m_val = f1_score(y_val_real, y_val_pred, average='weighted')
metrics_dict['val'].append(m_val)

m_val = accuracy_score(y_train_real, y_train_pred)
metrics_dict['train'].append(m_val)
m_val = accuracy_score(y_val_real, y_val_pred)
metrics_dict['val'].append(m_val)

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)

fig = ff.create_annotated_heatmap(conf_mtx_val, x=classes_names, y=classes_names, colorscale='Viridis')
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=0.5,
                        y=-0.1,
                        showarrow=False,
                        text="Predicted value",
                        xref="paper",
                        yref="paper"))
fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=-0.33,
                        y=0.5,
                        showarrow=False,
                        text="Real value",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))
fig.update_layout(margin=dict(t=50, l=200))
fig['data'][0]['showscale'] = True
save_figure(fig, "confusion_matrix_val_xgb_binary")

metrics_df = pd.DataFrame.from_dict(metrics_dict)
metrics_df.to_excel("figures/metrics_xgb_binary.xlsx", index=True)
