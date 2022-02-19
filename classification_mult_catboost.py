import numpy as np
import pandas as pd
from catboost import CatBoost
import plotly.figure_factory as ff
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from plot_evolution import plot_catboost_evolution


df = pd.read_excel('dataframes/dataframe.xlsx')
all_classes = pd.factorize(list(df['class']))[0]
all_features = df.iloc[:, :-1].to_numpy()
features_names = list(df.columns.values)[:-1]

ids_train, ids_val = train_test_split(np.arange(len(all_classes)), test_size=0.2, stratify=all_classes, random_state=7)

ids_train_test = []
for i in range(0, len(all_classes)):
    if i in ids_train:
        ids_train_test.append('train')
    elif i in ids_val:
        ids_train_test.append('test')
    else:
        print('Train test split error')

ids_train_test_df = pd.DataFrame(ids_train_test)
ids_train_test_df.to_excel("dataframes/train_test_split.xlsx", header=['status'], index=False)

train_features = all_features[ids_train, :]
val_features = all_features[ids_val, :]

train_classes = [all_classes[i] for i in list(ids_train)]
val_classes = [all_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi', 'im1', 'im2']

model_params = {'classes_count': 4,
                'loss_function': 'MultiClass',
                'learning_rate': 0.06,
                'depth': 6,
                'min_data_in_leaf': 1,
                'max_leaves': 31,
                'verbose': 1,
                'iterations': 10000,
                'early_stopping_rounds': 100}

model = CatBoost(params=model_params)
model.fit(train_features, train_classes, eval_set=(val_features, val_classes))
model.set_feature_names(features_names)
model.save_model(f"models/epoch_{model.best_iteration_}_cat.model")

train_pred = model.predict(train_features, prediction_type="Class")
val_pred = model.predict(val_features, prediction_type="Class")

y_train_real = train_classes
y_train_pred = [item[0] for item in train_pred]
y_val_real = val_classes
y_val_pred = [item[0] for item in val_pred]

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


def save_figure(fig, fn):
    Path(f"figures/lr({model_params['learning_rate']})_it({model_params['iterations']})").mkdir(parents=True,
                                                                                                exist_ok=True)
    fig.write_image(f"figures/lr({model_params['learning_rate']})_it({model_params['iterations']})/{fn}.png")
    fig.write_image(f"figures/lr({model_params['learning_rate']})_it({model_params['iterations']})/{fn}.pdf")


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
metrics_df.to_excel(f"figures/lr({model_params['learning_rate']})_it({model_params['iterations']})/metrics_cat.xlsx",
                    index=True)

plot_catboost_evolution(f"figures/lr({model_params['learning_rate']})_it({model_params['iterations']})/")
