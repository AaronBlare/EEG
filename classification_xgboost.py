import numpy as np
import pandas as pd
import xgboost as xgb
import plotly.figure_factory as ff
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
from plot_evolution import plot_xgboost_evolution


data_path = 'E:/YandexDisk/EEG/Dataframes/'
path = 'E:/YandexDisk/EEG/'
data_file = 'dataframe_1st_Day.xlsx'

classes = ['right_im2', 'left_im2', 'background']
suffix = f"1st_Day_{'_'.join(classes)}"

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

ids_train_val = []
for i in range(0, len(train_val_classes)):
    if i in ids_train:
        ids_train_val.append('train')
    elif i in ids_val:
        ids_train_val.append('val')
    else:
        print('Train test split error')

ids_train_val_df = pd.DataFrame(ids_train_val)
ids_train_val_df.to_excel(f"{data_path}train_test_split_{suffix}_xgboost.xlsx", header=['status'], index=False)

train_features = train_val_features[ids_train, :]
val_features = train_val_features[ids_val, :]

train_classes = [train_val_classes[i] for i in list(ids_train)]
val_classes = [train_val_classes[i] for i in list(ids_val)]

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=features_names)
dmat_val = xgb.DMatrix(val_features, val_classes, feature_names=features_names)
dmat_test = xgb.DMatrix(test_features, test_classes, feature_names=features_names)

if len(classes) == 2:
    model_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_split_loss': 0,
        'sampling_method': 'uniform',
        'eval_metric': 'logloss',
        'subsample': 1,
        'objective': 'binary:logistic',
        'verbosity': 1
    }
else:
    model_params = {
        'num_class': len(classes),
        'booster': 'gbtree',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_split_loss': 0,
        'sampling_method': 'uniform',
        'subsample': 1,
        'objective': 'multi:softprob',
        'verbosity': 1,
    }

progress = dict()

num_boost_round = 10000
early_stopping_rounds = 100
bst = xgb.train(
    params=model_params,
    dtrain=dmat_train,
    evals=[(dmat_train, "train"), (dmat_val, "val")],
    num_boost_round=num_boost_round,
    early_stopping_rounds=early_stopping_rounds,
    evals_result=progress
)
bst.save_model(f"{path}Models/{suffix}_epoch_{bst.best_iteration}_xgb.model")

train_pred = bst.predict(dmat_train)
val_pred = bst.predict(dmat_val)
test_pred = bst.predict(dmat_test)

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

metrics_dict = {'train': [], 'val': [], 'test': []}

metrics_dict['train'].append(f1_score(y_train_real, y_train_pred, average='weighted'))
metrics_dict['val'].append(f1_score(y_val_real, y_val_pred, average='weighted'))
metrics_dict['test'].append(f1_score(y_test_real, y_test_pred, average='weighted'))

metrics_dict['train'].append(accuracy_score(y_train_real, y_train_pred))
metrics_dict['val'].append(accuracy_score(y_val_real, y_val_pred))
metrics_dict['test'].append(accuracy_score(y_test_real, y_test_pred))

conf_mtx_train = confusion_matrix(y_train_real, y_train_pred)
conf_mtx_val = confusion_matrix(y_val_real, y_val_pred)
conf_mtx_test = confusion_matrix(y_test_real, y_test_pred)


def save_figure(fig, fn):
    Path(f"{path}Figures/{suffix}/xgboost/lr({model_params['learning_rate']})_it({num_boost_round})").mkdir(
        parents=True,
        exist_ok=True)
    fig.write_image(
        f"{path}Figures/{suffix}/xgboost/lr({model_params['learning_rate']})_it({num_boost_round})/{fn}.png")
    fig.write_image(
        f"{path}Figures/{suffix}/xgboost/lr({model_params['learning_rate']})_it({num_boost_round})/{fn}.pdf")


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
save_figure(fig, "confusion_matrix_val_xgb")

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
save_figure(fig, "confusion_matrix_test_xgb")

metrics_df = pd.DataFrame.from_dict(metrics_dict)
metrics_df.to_excel(
    f"{path}Figures/{suffix}/xgboost/lr({model_params['learning_rate']})_it({num_boost_round})/metrics_xgb.xlsx",
    index=True)

plot_xgboost_evolution(progress,
    f"{path}Figures/{suffix}/xgboost/lr({model_params['learning_rate']})_it({num_boost_round})/")
