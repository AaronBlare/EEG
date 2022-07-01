import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.decomposition import MiniBatchDictionaryLearning, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.graph_objects as go
import plotly.express as px
from scripts.plot_functions import plot_scatter_train_val


experiments = {'1st_day': ['train', 'val']}

for experiment in experiments:

    data_path = f'E:/YandexDisk/EEG/experiments/{experiment}/'

    data_df = pd.read_excel(f'{data_path}data.xlsx')
    features_df = pd.read_excel(f'{data_path}features_freq.xlsx')
    features = features_df['features'].tolist()

    data = data_df.loc[:, features].values
    subjects = data_df.loc[:, ['subject']].values
    num_subjects = np.unique(subjects).shape[0]
    classes = data_df.loc[:, ['class_simp']].values

    colors = px.colors.qualitative.Light24
    marker_symbols = ['circle', 'x']

    # PCA ==============================================================================================================

    pca = PCA(n_components=2, whiten=False)
    data_pca = pca.fit_transform(data)
    data_df['PC1'] = data_pca[:, 0]
    data_df['PC2'] = data_pca[:, 1]

    # Incremental PCA ==================================================================================================

    n_batches = 32
    ipca = IncrementalPCA(n_components=2)
    for data_batch in np.array_split(data, n_batches):
        ipca.partial_fit(data_batch)
    data_ipca = ipca.transform(data)
    data_df['IncrementalPC1'] = data_ipca[:, 0]
    data_df['IncrementalPC2'] = data_ipca[:, 1]

    # Kernel PCA =======================================================================================================

    kpca = KernelPCA(kernel='rbf', fit_inverse_transform=True, gamma=None, n_components=2)
    data_kpca = kpca.fit_transform(data)
    data_df['KernelPC1'] = data_kpca[:, 0]
    data_df['KernelPC2'] = data_kpca[:, 1]

    # Sparse PCA =======================================================================================================

    spca = SparsePCA(n_components=2, alpha=0.001)
    spca.fit(data)
    data_spca = spca.transform(data)
    data_df['SparsePC1'] = data_spca[:, 0]
    data_df['SparsePC2'] = data_spca[:, 1]

    # SVD ==============================================================================================================

    SVD_ = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5)
    SVD_.fit(data)
    data_svd = SVD_.transform(data)
    data_df['SVD1'] = data_svd[:, 0]
    data_df['SVD2'] = data_svd[:, 1]

    # GRP ==============================================================================================================

    GRP = GaussianRandomProjection(n_components=2, eps=0.5)
    GRP.fit(data)
    data_grp = GRP.transform(data)
    data_df['GaussianRandomProjection1'] = data_grp[:, 0]
    data_df['GaussianRandomProjection2'] = data_grp[:, 1]

    # SRP ==============================================================================================================

    SRP = SparseRandomProjection(n_components=2, density='auto', eps=0.5, dense_output=False)
    SRP.fit(data)
    data_srp = SRP.transform(data)
    data_df['SparseRandomProjection1'] = data_srp[:, 0]
    data_df['SparseRandomProjection2'] = data_srp[:, 1]

    # MDS ==============================================================================================================

    mds = MDS(n_components=2, metric=True)
    data_mds = mds.fit_transform(data)
    data_df['MultiDimensionalScale1'] = data_mds[:, 0]
    data_df['MultiDimensionalScale2'] = data_mds[:, 1]

    # ISOMAP ===========================================================================================================

    isomap = Isomap(n_components=2, n_neighbors=5)
    isomap.fit(data)
    data_isomap = isomap.transform(data)
    data_df['Isomap1'] = data_isomap[:, 0]
    data_df['Isomap2'] = data_isomap[:, 1]

    # MiniBatchDictionaryLearning ======================================================================================

    miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=2, batch_size=200, alpha=1, n_iter=25)
    miniBatchDictLearning.fit(data)
    data_batch = miniBatchDictLearning.fit_transform(data)
    data_df['MBDL1'] = data_batch[:, 0]
    data_df['MBDL2'] = data_batch[:, 1]

    # ICA ==============================================================================================================

    fastICA = FastICA(n_components=2, algorithm='parallel', whiten=True, tol=1e-3, max_iter=1000)
    data_ica = fastICA.fit_transform(data)
    data_df['IC1'] = data_ica[:, 0]
    data_df['IC2'] = data_ica[:, 1]

    # t-SNE ============================================================================================================

    tsne = TSNE(n_components=2, learning_rate=300, perplexity=30, early_exaggeration=12, init='random')
    data_tsne = tsne.fit_transform(data)
    data_df['tSNE1'] = data_tsne[:, 0]
    data_df['tSNE2'] = data_tsne[:, 1]

    # LLE ==============================================================================================================

    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, method='modified')
    lle.fit(data)
    data_lle = lle.transform(data)
    data_df['LLE1'] = data_lle[:, 0]
    data_df['LLE2'] = data_lle[:, 1]

    # LDA ==============================================================================================================

    lda = LinearDiscriminantAnalysis(n_components=2)
    data_lda = lda.fit(data, classes.ravel()).transform(data)
    data_df['LDA1'] = data_lda[:, 0]
    data_df['LDA2'] = data_lda[:, 1]

    methods_dict = {'PCA': ['PC1', 'PC2'],
                    'IncrementalPCA': ['IncrementalPC1', 'IncrementalPC2'],
                    'KernelPCA': ['KernelPC1', 'KernelPC2'],
                    'SparsePCA': ['SparsePC1', 'SparsePC2'],
                    'SingularValueDecomposition': ['SVD1', 'SVD2'],
                    'GaussianRandomProjection': ['GaussianRandomProjection1', 'GaussianRandomProjection2'],
                    'SparseRandomProjection': ['SparseRandomProjection1', 'SparseRandomProjection2'],
                    'MultiDimensionalScaling': ['MultiDimensionalScale1', 'MultiDimensionalScale2'],
                    'Isomap': ['Isomap1', 'Isomap2'],
                    'MiniBatchDictionaryLearning': ['MBDL1', 'MBDL2'],
                    'ICA': ['IC1', 'IC2'],
                    'T-SNE': ['tSNE1', 'tSNE2'],
                    'LocallyLinearEmbedding': ['LLE1', 'LLE2'],
                    'LinearDiscriminantAnalysis': ['LDA1', 'LDA2']}

    curr_experiment_name = '_'.join(experiments[experiment])
    save_path = f'E:/YandexDisk/EEG/Figures/dimensionality_reduction/{experiment}/{curr_experiment_name}/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for method in methods_dict:
        axis1 = methods_dict[method][0]
        axis2 = methods_dict[method][1]
        fig = go.Figure()
        plot_scatter_train_val(fig, experiments[experiment], marker_symbols, colors, data_df, axis1, axis2, method)
        fig.write_image(f"{save_path}{method}.png")
        fig.write_image(f"{save_path}{method}.pdf", format="pdf")
