import numpy as np
import pandas as pd
from catboost import CatBoost
import plotly.figure_factory as ff
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from plot_evolution import plot_catboost_evolution

data_path = 'E:/YandexDisk/EEG/Dataframes/'
path = 'E:/YandexDisk/EEG/'
data_file = 'dataframe_2nd_Day_sham.xlsx'

classes = ['right_im2', 'background']
suffix = f"sham_{'_'.join(classes)}"

df = pd.read_excel(data_path + data_file)
df_classes = df[df['class'].str[:].isin(classes)]

val_subjects = ['S5', 'S6', 'S7']
mask = df_classes['trial'].str[:2].isin(val_subjects)
df_val = df_classes[mask]
df_train_test = df_classes[~mask]

train_test_classes = pd.factorize(list(df_train_test['class']))[0]
train_test_features = df_train_test.iloc[:, 1:-1].to_numpy()

val_classes = pd.factorize(list(df_val['class']))[0]
val_features = df_val.iloc[:, 1:-1].to_numpy()

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

ids_train_test_df = pd.DataFrame(ids_train_test)
ids_train_test_df.to_excel(f"{data_path}train_test_split_{suffix}_catboost.xlsx", header=['status'], index=False)

train_features = train_test_features[ids_train, :]
test_features = train_test_features[ids_test, :]

train_classes = [train_test_classes[i] for i in list(ids_train)]
test_classes = [train_test_classes[i] for i in list(ids_test)]

if len(classes) == 2:
    model_params = {'loss_function': 'Logloss',
                    'learning_rate': 0.05,
                    'depth': 6,
                    'min_data_in_leaf': 1,
                    'max_leaves': 31,
                    'verbose': 1,
                    'iterations': 10000,
                    'early_stopping_rounds': 100}
else:
    model_params = {'classes_count': len(classes),
                    'loss_function': 'MultiClass',
                    'learning_rate': 0.06,
                    'depth': 6,
                    'min_data_in_leaf': 1,
                    'max_leaves': 31,
                    'verbose': 1,
                    'iterations': 10000,
                    'early_stopping_rounds': 100}

model = CatBoost(params=model_params)
model.fit(train_features, train_classes, eval_set=(test_features, test_classes))
model.set_feature_names(features_names)
features_importances = pd.DataFrame.from_dict(
    {'feature': model.feature_names_, 'importance': list(model.feature_importances_)})
model.save_model(f"{path}Models/{suffix}_epoch_{model.best_iteration_}_cat.model")

train_pred = model.predict(train_features, prediction_type="Class")
test_pred = model.predict(test_features, prediction_type="Class")
val_pred = model.predict(val_features, prediction_type="Class")

y_train_real = train_classes
y_train_pred = train_pred
y_test_real = test_classes
y_test_pred = test_pred
y_val_real = val_classes
y_val_pred = val_pred

metrics_dict = {'train': [], 'test': [], 'val': []}

metrics_dict['train'].append(f1_score(y_train_real, y_train_pred, average='weighted'))
metrics_dict['test'].append(f1_score(y_test_real, y_test_pred, average='weighted'))
metrics_dict['val'].append(f1_score(y_val_real, y_val_pred, average='weighted'))

metrics_dict['train'].append(accuracy_score(y_train_real, y_train_pred))
metrics_dict['test'].append(accuracy_score(y_test_real, y_test_pred))
metrics_dict['val'].append(accuracy_score(y_val_real, y_val_pred))

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_test = confusion_matrix(y_test_real, y_test_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)


def save_figure(fig, fn):
    Path(f"{path}Figures/{suffix}/catboost/lr({model_params['learning_rate']})_it({model_params['iterations']})").mkdir(
        parents=True,
        exist_ok=True)
    fig.write_image(
        f"{path}Figures/{suffix}/catboost/lr({model_params['learning_rate']})_it({model_params['iterations']})/{fn}.png")
    fig.write_image(
        f"{path}Figures/{suffix}/catboost/lr({model_params['learning_rate']})_it({model_params['iterations']})/{fn}.pdf")


fig = ff.create_annotated_heatmap(conf_mtx_test, x=classes, y=classes, colorscale='Viridis')
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
save_figure(fig, "confusion_matrix_test_cat")

fig = ff.create_annotated_heatmap(conf_mtx_val, x=classes, y=classes, colorscale='Viridis')
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
metrics_df.to_excel(
    f"{path}Figures/{suffix}/catboost/lr({model_params['learning_rate']})_it({model_params['iterations']})/metrics_cat.xlsx",
    index=True)

plot_catboost_evolution(f"{path}Figures/{suffix}/catboost/lr({model_params['learning_rate']})_it({model_params['iterations']})/")
