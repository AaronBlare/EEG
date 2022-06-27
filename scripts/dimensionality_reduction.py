import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import plotly.graph_objects as go
import plotly.express as px
from scripts.plot_functions import add_scatter_trace, add_layout


experiment_name = '1st_day'
experiment_types = [['right_real', 'left_real'],
                    ['right_quasi', 'left_quasi'],
                    ['right_im', 'left_im']]

data_path = f'E:/YandexDisk/EEG/experiments/{experiment_name}/'
save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/'
Path(save_path).mkdir(parents=True, exist_ok=True)

data_df = pd.read_excel(f'{data_path}data.xlsx')
features_df = pd.read_excel(f'{data_path}features_freq.xlsx')
features = features_df['features'].tolist()

data = data_df.loc[:, features].values
subjects = data_df.loc[:, ['subject']].values
num_subjects = np.unique(subjects).shape[0]
classes = data_df.loc[:, ['class_simp']].values
# data = StandardScaler().fit_transform(data)
# data = pd.DataFrame(data)

# PCA ==================================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/PCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

pca = PCA()
data_pca = pca.fit_transform(data)
data_pca = pd.DataFrame(data_pca[:, :2])
data_pca['subject'] = subjects
data_pca['class'] = classes
pca_columns = ["PC1", "PC2", 'subject', 'class']
data_pca.columns = pca_columns

marker_symbols = ['circle', 'x']
for experiment in experiment_types:
    fig = go.Figure()
    for movement_id in range(0, len(experiment)):
        movement = experiment[movement_id]
        symbol = marker_symbols[movement_id]
        curr_movement_data = data_pca.loc[data_pca['class'] == movement]
        for subject_id in range(0, num_subjects):
            curr_subject_data = curr_movement_data.loc[curr_movement_data['subject'] == f'S{subject_id}']
            add_scatter_trace(fig,
                              curr_subject_data['PC1'].values,
                              curr_subject_data['PC2'].values,
                              f'S{subject_id}_{movement}',
                              symbol)
    add_layout(fig, f'PC1', f'PC2', f'')
    colors = px.colors.sample_colorscale("turbo", [col / (num_subjects - 1) for col in range(num_subjects)])
    colors.extend(colors)
    fig.update_layout({'colorway': colors})
    fig.update_layout(legend_font_size=13)
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    for movement_id in range(0, len(experiment)):
        movement = experiment[movement_id]
        symbol = marker_symbols[0]
        curr_movement_data = data_pca.loc[data_pca['class'] == movement]
        add_scatter_trace(fig, curr_movement_data['PC1'].values, curr_movement_data['PC2'].values, movement, symbol)
    add_layout(fig, f'PC1', f'PC2', f'')
    fig.update_layout({'colorway': px.colors.qualitative.D3})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# Kernel PCA ===========================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/KernelPCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=None)
data_kpca = kpca.fit_transform(data)
data_kpca = pd.DataFrame(data_kpca[:, :2])
data_kpca['subject'] = subjects
data_kpca['class'] = classes
kpca_columns = ["PC1", "PC2", 'subject', 'class']
data_kpca.columns = kpca_columns

marker_symbols = ['circle', 'x']
for experiment in experiment_types:
    fig = go.Figure()
    for movement_id in range(0, len(experiment)):
        movement = experiment[movement_id]
        symbol = marker_symbols[movement_id]
        curr_movement_data = data_kpca.loc[data_kpca['class'] == movement]
        for subject_id in range(0, num_subjects):
            curr_subject_data = curr_movement_data.loc[curr_movement_data['subject'] == f'S{subject_id}']
            add_scatter_trace(fig,
                              curr_subject_data['PC1'].values,
                              curr_subject_data['PC2'].values,
                              f'S{subject_id}_{movement}',
                              symbol)
    add_layout(fig, f'PC1', f'PC2', f'')
    colors = px.colors.sample_colorscale("turbo", [col / (num_subjects - 1) for col in range(num_subjects)])
    colors.extend(colors)
    fig.update_layout({'colorway': colors})
    fig.update_layout(legend_font_size=13)
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    for movement_id in range(0, len(experiment)):
        movement = experiment[movement_id]
        symbol = marker_symbols[0]
        curr_movement_data = data_kpca.loc[data_kpca['class'] == movement]
        add_scatter_trace(fig, curr_movement_data['PC1'].values, curr_movement_data['PC2'].values, movement, symbol)
    add_layout(fig, f'PC1', f'PC2', f'')
    fig.update_layout({'colorway': px.colors.qualitative.D3})
    fig.update_layout(legend_font_size=20)
    fig.update_layout(margin=go.layout.Margin(l=90, r=20, b=80, t=65, pad=0))
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

