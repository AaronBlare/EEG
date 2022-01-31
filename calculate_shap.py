import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import shap
import matplotlib.pyplot as plt
from pathlib import Path


shap.initjs()

train_test_split_df = pd.read_excel('dataframes/train_test_split.xlsx')
train_test_split = train_test_split_df['status'].to_list()
ids_train = [item[0] for item in enumerate(train_test_split) if item[1] == 'train']

df = pd.read_excel('dataframes/dataframe.xlsx')
all_classes = pd.factorize(list(df['class']))[0]
classes_names = set(list(df['class']))
all_features = df.iloc[:, :-1].to_numpy()
all_names = list(df.columns.values)[:-1]
train_features = all_features[ids_train, :]

from_file = CatBoostClassifier()
model = from_file.load_model("models/epoch_9904_cat.model")


def proba(X):
    y = model.predict(X, prediction_type='Probability')
    return y


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_features)

shap.summary_plot(
    shap_values=shap_values,
    features=train_features,
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
        features=train_features,
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
                data=train_features[m_id],
                feature_names=all_names
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

for st_id, st in enumerate(classes_names):
    class_shap_values = shap_values[st_id]
    d = {'epochs': ids_val}
    for f_id in range(0, len(train_features[0, :])):
        curr_shap = class_shap_values[:, f_id]
        feature_name = all_names[f_id]
        d[f"{feature_name}_shap"] = curr_shap
    df_features = pd.DataFrame(d)
    df_features.to_excel(f"{st}/shap.xlsx", index=False)
