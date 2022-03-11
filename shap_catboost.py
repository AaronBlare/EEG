import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
model_file = [f"{path}Models/{filename}" for filename in os.listdir(f"{path}Models/") if filename.startswith(suffix)]

df = pd.read_excel(data_path + data_file)
df_classes = df[df['class'].str[:].isin(classes)]

val_subjects = ['S5', 'S6', 'S7']
mask = df_classes['trial'].str[:2].isin(val_subjects)
df_val = df_classes[mask]
df_train_test = df_classes[~mask]

train_test_classes = pd.factorize(list(df_train_test['class']))[0]
train_test_features = df_train_test.iloc[:, 1:-1].to_numpy()

features_names = list(df_train_test.columns.values)[1:-1]

ids_train, ids_test = train_test_split(np.arange(len(train_test_classes)), test_size=0.2,
                                       stratify=train_test_classes, random_state=7)

ids_train_test = []
for i in range(0, len(train_test_classes)):
    if i in ids_train:
        ids_train_test.append('train')
    elif i in ids_test:
        ids_train_test.append('test')
    else:
        print('Train test split error')

train_features = train_test_features[ids_train, :]
test_features = train_test_features[ids_test, :]

train_classes = [train_test_classes[i] for i in list(ids_train)]
test_classes = [train_test_classes[i] for i in list(ids_test)]

from_file = CatBoostClassifier()
model = from_file.load_model(model_file[0])


def proba(X):
    y = model.predict(X, prediction_type='Probability')
    return y


if len(classes) == 2:
    explainer = shap.TreeExplainer(model, data=train_features, model_output='probability')
    shap_values = explainer.shap_values(train_features)
else:
    explainer = shap.KernelExplainer(proba, data=train_features)
    shap_values = explainer.shap_values(train_features)

probs = model.predict(test_features, prediction_type='Probability')
for per_id in range(0, len(ids_val)):
    for st_id, st in enumerate(classes):
        probs_real = probs[per_id, st_id]
        probs_expl = explainer.expected_value[st_id] + sum(shap_values[st_id][per_id])
        if abs(probs_real - probs_expl) > 1e-6:
            print(f"diff between prediction: {abs(probs_real - probs_expl)}")

for st_id, st in enumerate(classes):
    class_shap_values = shap_values[st_id]
    d = {'subjects': ids_test}
    for f_id in range(0, len(test_features[0, :])):
        curr_shap = class_shap_values[:, f_id]
        feature_name = features_names[f_id]
        d[f"{feature_name}_shap"] = curr_shap
    df_features = pd.DataFrame(d)
    Path(f"{path}Shap/{suffix}/{st}").mkdir(parents=True, exist_ok=True)
    df_features.to_excel(f"{path}Shap/{suffix}/{st}/shap.xlsx", index=False)

shap.summary_plot(
    shap_values=shap_values,
    features=test_features,
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

for st_id, st in enumerate(classes):
    shap.summary_plot(
        shap_values=shap_values[st_id],
        features=test_features,
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

train_pred = model.predict(train_features, prediction_type="Class")
test_pred = model.predict(test_features, prediction_type="Class")
test_pred_probs = model.predict(test_features, prediction_type="Probability")

y_train_real = train_classes
y_train_pred = [item[0] for item in train_pred]
y_test_real = val_classes
y_test_pred = [item[0] for item in test_pred]

is_correct_pred = (np.array(y_test_real) == np.array(y_test_pred))
mistakes_ids = np.where(is_correct_pred == False)[0]

metrics_dict = {'train': [], 'test': []}

metrics_dict['train'].append(f1_score(y_train_real, y_train_pred, average='weighted'))
metrics_dict['test'].append(f1_score(y_test_real, y_test_pred, average='weighted'))

metrics_dict['train'].append(accuracy_score(y_train_real, y_train_pred))
metrics_dict['test'].append(accuracy_score(y_test_real, y_test_pred))

for m_id in mistakes_ids:
    subj_cl = y_test_real[m_id]
    subj_pred_cl = y_test_pred[m_id]
    for st_id, st in enumerate(classes):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[st_id][m_id],
                base_values=explainer.expected_value[st_id],
                data=test_features[m_id],
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
for subj_id in range(test_features.shape[0]):
    subj_cl = y_test_real[subj_id]
    if passed_examples[subj_cl] < len(y_train_real):
        for st_id, st in enumerate(classes):
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[st_id][subj_id],
                    base_values=explainer.expected_value[st_id],
                    data=test_features[subj_id],
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
