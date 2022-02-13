import copy
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path


shap.initjs()

train_test_split_df = pd.read_excel('dataframes/train_test_split_2.xlsx')
train_test_split = train_test_split_df['status'].to_list()
ids_train = [item[0] for item in enumerate(train_test_split) if item[1] == 'train']

df = pd.read_excel('dataframes/dataframe_2.xlsx')
all_classes = pd.factorize(list(df['class']))[0]
train_classes = [all_classes[i] for i in list(ids_train)]
classes_names = ['real', 'quasi', 'im1', 'im2']
all_features = df.iloc[:, :-1].to_numpy()
features_names = list(df.columns.values)[:-1]
train_features = all_features[ids_train, :]

model = xgb.Booster()
model.load_model("models/epoch_217_xgb.model")

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=features_names)
probs = model.predict(dmat_train)

explainer = shap.TreeExplainer(model, data=train_features, model_output='probability')
shap_values = explainer.shap_values(train_features)

base_probability = []
base_probability_numerator = []
base_probability_denominator = 0
for class_id in range(0, len(explainer.expected_value)):
    base_probability_numerator.append(np.exp(explainer.expected_value[class_id]))
    base_probability_denominator += np.exp(explainer.expected_value[class_id])
for class_id in range(0, len(explainer.expected_value)):
    base_probability.append(base_probability_numerator[class_id] / base_probability_denominator)

delta_prob_subject_class = []
for class_id in range(0, len(explainer.expected_value)):
    delta_prob_subject_class.append([])
    for subject_id in range(0, len(ids_train)):
        real_prob = probs[subject_id, class_id]
        expl_prob_numerator = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
        expl_prob_denominator = 0
        for c_id in range(0, len(explainer.expected_value)):
            expl_prob_denominator += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
        expl_prob = expl_prob_numerator / expl_prob_denominator
        delta_prob_subject_class[class_id].append(expl_prob - base_probability[class_id])
        if abs(real_prob - expl_prob) > 1e-6:
            print(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(real_prob - expl_prob)}")

shap_values_prob = copy.deepcopy(shap_values)
for class_id in range(0, len(explainer.expected_value)):
    for subject_id in range(0, len(ids_train)):
        for feature_id in range(0, len(features_names)):
            shap_probability_numerator = np.exp(explainer.expected_value[class_id] + shap_values[class_id][subject_id, feature_id])
            shap_probability_denominator = 0
            for c_id in range(0, len(explainer.expected_value)):
                shap_probability_denominator += np.exp(explainer.expected_value[c_id] + shap_values[c_id][subject_id, feature_id])
            shap_values_prob[class_id][subject_id, feature_id] = (shap_probability_numerator / shap_probability_denominator) - base_probability[class_id]

for class_id in range(0, len(explainer.expected_value)):
    for subject_id in range(0, len(ids_train)):
        real_shap_contrib = delta_prob_subject_class[class_id][subject_id]
        expl_shap_contrib = sum(shap_values_prob[class_id][subject_id])
        if abs(real_shap_contrib - expl_shap_contrib) > 1e-6:
            print(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {abs(real_shap_contrib - expl_shap_contrib)}")

shap_catboost = model.get_feature_importance(Pool(train_features, train_classes), type='ShapValues')



explainer_ker = shap.KernelExplainer(proba, data=train_features)
shap_values_ker = explainer_ker.shap_values(train_features)

probs = model.predict(train_features, prediction_type='Probability')
for per_id in range(0, df.shape[0]):
    for st_id, st in enumerate(classes_names):
        probs_real = probs[per_id, st_id]
        probs_expl = explainer_ker.expected_value[st_id] + sum(shap_values_ker[st_id][per_id])
        if abs(probs_real - probs_expl) > 1e-6:
            print(f"diff between prediction: {abs(probs_real - probs_expl)}")

shap.summary_plot(
    shap_values=shap_values,
    features=train_features,
    feature_names=features_names,
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
        features=train_features,
        feature_names=features_names,
        max_display=30,
        plot_size=(18, 10),
        plot_type="violin",
        title=st,
        show=False
    )
    plt.savefig(f"figures/{st}_beeswarm.png")
    plt.savefig(f"figures/{st}_beeswarm.pdf")
    plt.close()

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

m_val = accuracy_score(y_train_real, y_train_pred)
metrics_dict['train'].append(m_val)
m_val = accuracy_score(y_val_real, y_val_pred)
metrics_dict['val'].append(m_val)

for m_id in mistakes_ids:
    subj_cl = y_val_real[m_id]
    subj_pred_cl = y_val_pred[m_id]
    for st_id, st in enumerate(classes_names):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[st_id][m_id],
                base_values=explainer.expected_value[st_id],
                data=train_features[m_id],
                feature_names=features_names
            ),
            max_display=30,
            show=False
        )
        fig = plt.gcf()
        fig.set_size_inches(18, 10, forward=True)
        Path(f"figures/errors/real({classes_names[subj_cl]})_pred({classes_names[subj_pred_cl]})/{m_id}").mkdir(
            parents=True,
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
                    data=train_features[subj_id],
                    feature_names=features_names
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

for st_id, st in enumerate(classes_names):
    class_shap_values = shap_values[st_id]
    d = {'epochs': ids_val}
    for f_id in range(0, len(train_features[0, :])):
        curr_shap = class_shap_values[:, f_id]
        feature_name = features_names[f_id]
        d[f"{feature_name}_shap"] = curr_shap
    df_features = pd.DataFrame(d)
    df_features.to_excel(f"{st}/shap.xlsx", index=False)
