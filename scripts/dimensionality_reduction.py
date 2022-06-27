import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import plotly.graph_objects as go
import plotly.express as px
from scripts.plot_functions import plot_scatter_by_subject, plot_scatter

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

colors = px.colors.sample_colorscale("turbo", [col / (num_subjects - 1) for col in range(num_subjects)])
marker_symbols = ['circle', 'x']

# PCA ==================================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/PCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

pca = PCA(n_components=320, whiten=False)
data_pca = pca.fit_transform(data)
data_pca = pd.DataFrame(data_pca[:, :2])
data_pca['subject'] = subjects
data_pca['class'] = classes
pca_columns = ["PC1", "PC2", 'subject', 'class']
data_pca.columns = pca_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_pca, num_subjects,
                            'PC1', 'PC2', 'PCA')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_pca, 'PC1', 'PC2', 'PCA')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# Incremental PCA ======================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/IncrementalPCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

n_batches = 32
ipca = IncrementalPCA(n_components=8)
for data_batch in np.array_split(data, n_batches):
    ipca.partial_fit(data_batch)
data_ipca = ipca.transform(data)
data_ipca = pd.DataFrame(data_ipca[:, :2])
data_ipca['subject'] = subjects
data_ipca['class'] = classes
ipca_columns = ["PC1", "PC2", 'subject', 'class']
data_ipca.columns = ipca_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_ipca, num_subjects,
                            'PC1', 'PC2', 'IncrementalPCA')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_ipca, 'PC1', 'PC2', 'IncrementalPCA')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# Kernel PCA ===========================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/KernelPCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=None, n_components=320)
data_kpca = kpca.fit_transform(data)
data_kpca = pd.DataFrame(data_kpca[:, :2])
data_kpca['subject'] = subjects
data_kpca['class'] = classes
kpca_columns = ["PC1", "PC2", 'subject', 'class']
data_kpca.columns = kpca_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_kpca, num_subjects,
                            'PC1', 'PC2', 'KernelPCA')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_kpca, 'PC1', 'PC2', 'KernelPCA')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# Sparse PCA ===========================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/SparsePCA/'
Path(save_path).mkdir(parents=True, exist_ok=True)

spca = SparsePCA(n_components=10, alpha=0.001)
spca.fit(data)
data_spca = spca.transform(data)
data_spca = pd.DataFrame(data_spca[:, :2])
data_spca['subject'] = subjects
data_spca['class'] = classes
spca_columns = ["PC1", "PC2", 'subject', 'class']
data_spca.columns = spca_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_spca, num_subjects,
                            'PC1', 'PC2', 'SparcePCA')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_spca, 'PC1', 'PC2', 'SparsePCA')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# SVD ==================================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/SVD/'
Path(save_path).mkdir(parents=True, exist_ok=True)

SVD_ = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5)
SVD_.fit(data)
data_svd = SVD_.transform(data)
data_svd = pd.DataFrame(data_svd[:, :2])
data_svd['subject'] = subjects
data_svd['class'] = classes
svd_columns = ["SVD1", "SVD2", 'subject', 'class']
data_svd.columns = svd_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_svd, num_subjects,
                            'SVD1', 'SVD2', 'SVD')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_svd, 'SVD1', 'SVD2', 'SVD')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# GRP ==================================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/GRP/'
Path(save_path).mkdir(parents=True, exist_ok=True)

GRP = GaussianRandomProjection(n_components=100, eps=0.5)
GRP.fit(data)
data_grp = GRP.transform(data)
data_grp = pd.DataFrame(data_grp[:, :2])
data_grp['subject'] = subjects
data_grp['class'] = classes
grp_columns = ["GRP1", "GRP2", 'subject', 'class']
data_grp.columns = grp_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_grp, num_subjects,
                            'GRP1', 'GRP2', 'GRP')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_grp, 'GRP1', 'GRP2', 'GRP')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")

# SRP ==================================================================================================================

save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment_name}/SRP/'
Path(save_path).mkdir(parents=True, exist_ok=True)

SRP = SparseRandomProjection(n_components=100, density='auto', eps=0.5, dense_output=False)
SRP.fit(data)
data_srp = SRP.transform(data)
data_srp = pd.DataFrame(data_srp[:, :2])
data_srp['subject'] = subjects
data_srp['class'] = classes
srp_columns = ["SRP1", "SRP2", 'subject', 'class']
data_srp.columns = srp_columns

for experiment in experiment_types:
    fig = go.Figure()
    plot_scatter_by_subject(fig, experiment, marker_symbols, colors, data_srp, num_subjects,
                            'SRP1', 'SRP2', 'SRP')
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}subject_{'_'.join(experiment)}.pdf", format="pdf")

    fig = go.Figure()
    plot_scatter(fig, experiment, marker_symbols, data_srp, 'SRP1', 'SRP2', 'SRP')
    fig.write_image(f"{save_path}{'_'.join(experiment)}.png")
    fig.write_image(f"{save_path}{'_'.join(experiment)}.pdf", format="pdf")
