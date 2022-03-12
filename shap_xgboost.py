import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import os

shap.initjs()

data_path = 'E:/YandexDisk/EEG/Dataframes/'
path = 'E:/YandexDisk/EEG/'
data_file = 'dataframe_2nd_Day_TMS.xlsx'

classes = ['right_im1', 'background']
suffix = f"TMS_{'_'.join(classes)}"
model_file = [f"{path}Models/{filename}" for filename in os.listdir(f"{path}Models/") if
              filename.startswith(suffix) and filename.endswith('xgb.model')]

df = pd.read_excel(data_path + data_file)
df_classes = df[df['class'].str[:].isin(classes)]

test_subjects = ['S5', 'S6', 'S7', 'S8']
mask = df_classes['trial'].str[:2].isin(test_subjects)
df_test = df_classes[mask]
df_train_val = df_classes[~mask]

train_val_classes = pd.factorize(list(df_train_val['class']))[0]
train_val_features = df_train_val.iloc[:, 1:-1].to_numpy()

test_classes = pd.factorize(list(df_test['class']))[0]
test_features = df_test.iloc[:, 1:-1].to_numpy()

features_names = list(df_train_val.columns.values)[1:-1]

ids_train, ids_val = train_test_split(np.arange(len(train_val_classes)), test_size=0.2,
                                      stratify=train_val_classes, random_state=322)

train_features = train_val_features[ids_train, :]
val_features = train_val_features[ids_val, :]

train_classes = [train_val_classes[i] for i in list(ids_train)]
val_classes = [train_val_classes[i] for i in list(ids_val)]

model = xgb.Booster()
model.load_model(model_file[0])

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=features_names)
dmat_val = xgb.DMatrix(val_features, val_classes, feature_names=features_names)
dmat_test = xgb.DMatrix(test_features, test_classes, feature_names=features_names)

probs = model.predict(dmat_val)

explainer = shap.TreeExplainer(model, data=val_features, model_output='probability')
shap_values = explainer.shap_values(val_features)

for per_id in range(0, len(ids_val)):
    for st_id, st in enumerate(classes):
        if len(classes) == 2:
            probs_real = probs[per_id]
            probs_expl = explainer.expected_value + sum(shap_values[per_id])
            if abs(probs_real - probs_expl) > 1e-6:
                print(f"diff between prediction: {abs(probs_real - probs_expl)}")
        else:
            probs_real = probs[per_id, st_id]
            probs_expl = explainer.expected_value[st_id] + sum(shap_values[st_id][per_id])
            if abs(probs_real - probs_expl) > 1e-6:
                print(f"diff between prediction: {abs(probs_real - probs_expl)}")

for st_id, st in enumerate(classes):
    d = {'subjects': ids_val}
    if len(classes) == 2:
        class_shap_values = shap_values
    else:
        class_shap_values = shap_values[st_id]
    for f_id in range(0, len(val_features[0, :])):
        curr_shap = class_shap_values[:, f_id]
        feature_name = features_names[f_id]
        d[f"{feature_name}_shap"] = curr_shap
    df_features = pd.DataFrame(d)
    Path(f"{path}Shap/{suffix}/{st}").mkdir(parents=True, exist_ok=True)
    df_features.to_excel(f"{path}Shap/{suffix}/{st}/shap.xlsx", index=False)

shap.summary_plot(
    shap_values=shap_values,
    features=val_features,
    feature_names=features_names,
    max_display=30,
    class_names=classes,
    class_inds=list(range(len(classes))),
    plot_size=(18, 10),
    show=False,
    color=plt.get_cmap("Set1")
)
plt.savefig(f"{path}Shap/{suffix}/SHAP_bar.png")
plt.savefig(f"{path}Shap/{suffix}/SHAP_bar.pdf")
plt.close()

if len(classes) > 2:
    for st_id, st in enumerate(classes):
        shap.summary_plot(
            shap_values=shap_values[st_id],
            features=val_features,
            feature_names=features_names,
            max_display=30,
            plot_size=(18, 10),
            plot_type="violin",
            title=st,
            show=False
        )
        plt.savefig(f"{path}Shap/{suffix}/{st}_beeswarm.png")
        plt.savefig(f"{path}Shap/{suffix}/{st}_beeswarm.pdf")
        plt.close()

train_pred = model.predict(dmat_train)
val_pred = model.predict(dmat_val)
test_pred = model.predict(dmat_test)

y_train_real = train_classes
y_val_real = val_classes
y_test_real = test_classes
if len(classes) > 2:
    y_train_pred = np.argmax(train_pred, 1)
    y_val_pred = np.argmax(val_pred, 1)
    y_test_pred = np.argmax(test_pred, 1)
else:
    y_train_pred = np.where(train_pred > 0.5, 1, 0)
    y_val_pred = np.where(val_pred > 0.5, 1, 0)
    y_test_pred = np.where(test_pred > 0.5, 1, 0)

is_correct_pred = (np.array(y_val_real) == np.array(y_val_pred))
mistakes_ids = np.where(is_correct_pred == False)[0]

metrics_dict = {'train': [], 'val': [], 'test': []}

metrics_dict['train'].append(f1_score(y_train_real, y_train_pred, average='weighted'))
metrics_dict['val'].append(f1_score(y_val_real, y_val_pred, average='weighted'))
metrics_dict['test'].append(f1_score(y_test_real, y_test_pred, average='weighted'))

metrics_dict['train'].append(accuracy_score(y_train_real, y_train_pred))
metrics_dict['val'].append(accuracy_score(y_val_real, y_val_pred))
metrics_dict['test'].append(accuracy_score(y_test_real, y_test_pred))

for m_id in mistakes_ids:
    subj_cl = y_val_real[m_id]
    subj_pred_cl = y_val_pred[m_id]
    for st_id, st in enumerate(classes):
        if len(classes) > 2:
            curr_shap = shap_values[st_id][m_id]
            base_val = explainer.expected_value[st_id]
        else:
            curr_shap = shap_values[m_id]
            base_val = explainer.expected_value
        shap.waterfall_plot(
            shap.Explanation(
                values=curr_shap,
                base_values=base_val,
                data=val_features[m_id],
                feature_names=features_names
            ),
            max_display=30,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(18, 10, forward=True)
        Path(
            f"{path}Shap/{suffix}/errors/real({classes[subj_cl]})_pred({classes[subj_pred_cl]})/{m_id}").mkdir(
            parents=True,
            exist_ok=True)
        fig.savefig(
            f"{path}Shap/{suffix}/errors/real({classes[subj_cl]})_pred({classes[subj_pred_cl]})/{m_id}/waterfall_{st}.pdf")
        fig.savefig(
            f"{path}Shap/{suffix}/errors/real({classes[subj_cl]})_pred({classes[subj_pred_cl]})/{m_id}/waterfall_{st}.png")
        plt.close()

passed_examples = {x: 0 for x in range(len(classes))}
for subj_id in range(val_features.shape[0]):
    subj_cl = y_val_real[subj_id]
    if passed_examples[subj_cl] < len(y_train_real):
        for st_id, st in enumerate(classes):
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][subj_id],
                    base_values=explainer.expected_value[st_id],
                    data=val_features[subj_id],
                    feature_names=features_names
                ),
                max_display=30,
                show=False
            )
            fig = plt.gcf()
            fig.set_size_inches(18, 10, forward=True)
            Path(f"{path}Shap/{suffix}/figures/{classes[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}").mkdir(
                parents=True,
                exist_ok=True)
            fig.savefig(
                f"{path}Shap/{suffix}/figures/{classes[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.pdf")
            fig.savefig(
                f"{path}Shap/{suffix}/figures/{classes[subj_cl]}/{passed_examples[subj_cl]}_{subj_id}/waterfall_{st}.png")
            plt.close()
        passed_examples[subj_cl] += 1
