import xgboost as xgb
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from plot_evolution import plot_xgboost_evolution


df = pd.read_excel('dataframes/dataframe_2.xlsx')
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
ids_train_test_df.to_excel("dataframes/train_test_split_2.xlsx", header=['status'], index=False)

train_features = all_features[ids_train, :]
val_features = all_features[ids_val, :]

train_classes = [all_classes[i] for i in list(ids_train)]
val_classes = [all_classes[i] for i in list(ids_val)]

classes_names = ['real', 'quasi', 'im1', 'im2']

dmat_train = xgb.DMatrix(train_features, train_classes, feature_names=features_names)
dmat_val = xgb.DMatrix(val_features, val_classes, feature_names=features_names)

model_params = {
    'num_class': 4,
    'booster': 'gbtree',
    'learning_rate': 0.005,
    'max_depth': 6,
    'min_split_loss': 0,
    'sampling_method': 'uniform',
    'subsample': 1,
    'objective': 'multi:softprob',
    'verbosity': 1,
}

progress = dict()

num_boost_round = 5000
early_stopping_rounds = 100
bst = xgb.train(
    params=model_params,
    dtrain=dmat_train,
    evals=[(dmat_train, "train"), (dmat_val, "val")],
    num_boost_round=num_boost_round,
    early_stopping_rounds=early_stopping_rounds,
    evals_result=progress
)
bst.save_model(f"models/epoch_{bst.best_iteration}_xgb.model")

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

def save_figure(fig, fn):
    Path(f"figures/lr({model_params['learning_rate']})_it({num_boost_round})").mkdir(parents=True, exist_ok=True)
    fig.write_image(f"figures/lr({model_params['learning_rate']})_it({num_boost_round})/{fn}.png")
    fig.write_image(f"figures/lr({model_params['learning_rate']})_it({num_boost_round})/{fn}.pdf")


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
metrics_df.to_excel(f"figures/lr({model_params['learning_rate']})_it({num_boost_round})/metrics_xgb.xlsx",
                    index=True)

plot_xgboost_evolution(progress, f"figures/lr({model_params['learning_rate']})_it({num_boost_round})/")
