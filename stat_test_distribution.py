import numpy as np
import mne
from mne.externals.pymatreader import read_mat
import pandas as pd
import scipy
from scipy.stats import kruskal, mannwhitneyu
import statsmodels
from statsmodels.stats.multitest import multipletests
import os

data_path = 'E:/YandexDisk/EEG/Data/'
data_files = ['2nd_Day_TMS.mat', '2nd_Day_TMS_5new_subj.mat']

dataframe_path = 'E:/YandexDisk/EEG/Dataframes/'
data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}.xlsx"

dataframe_stat_path = 'E:/YandexDisk/EEG/Stat/'

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    mannwhitney_TMS_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            res = mannwhitneyu(curr_im1, curr_im2)
            pval = res.pvalue
            if not np.isnan(pval):
                mannwhitney_TMS_im1_im2['pval'].append(pval)
                mannwhitney_TMS_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(mannwhitney_TMS_im1_im2['pval'], method='fdr_bh')
    mannwhitney_TMS_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(mannwhitney_TMS_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/mannwhitney_TMS_im1_im2.xlsx", header=True, index=False)


data_files = ['DAY2_SHAM.mat']

data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}.xlsx"

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    mannwhitney_sham_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            res = mannwhitneyu(curr_im1, curr_im2)
            pval = res.pvalue
            if not np.isnan(pval):
                mannwhitney_sham_im1_im2['pval'].append(pval)
                mannwhitney_sham_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(mannwhitney_sham_im1_im2['pval'], method='fdr_bh')
    mannwhitney_sham_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(mannwhitney_sham_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/mannwhitney_sham_im1_im2.xlsx", header=True, index=False)


data_files = ['1st_Day.mat']

data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}.xlsx"

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    mannwhitney_1st_day_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            res = mannwhitneyu(curr_im1, curr_im2)
            pval = res.pvalue
            if not np.isnan(pval):
                mannwhitney_1st_day_im1_im2['pval'].append(pval)
                mannwhitney_1st_day_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(mannwhitney_1st_day_im1_im2['pval'], method='fdr_bh')
    mannwhitney_1st_day_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(mannwhitney_1st_day_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/mannwhitney_1st_day_im1_im2.xlsx", header=True, index=False)


data_files = ['2nd_Day_TMS.mat', '2nd_Day_TMS_5new_subj.mat']

data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}_background.xlsx"

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_real = df.loc[df['class'] == 'right_real']
    df_quasi = df.loc[df['class'] == 'right_quasi']
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    kruskal_TMS_real_quasi_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_real = df_real[column]
            curr_quasi = df_quasi[column]
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            stat, pval = kruskal(curr_real, curr_im1, curr_im1, curr_im2)
            if not np.isnan(pval):
                kruskal_TMS_real_quasi_im1_im2['pval'].append(pval)
                kruskal_TMS_real_quasi_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_TMS_real_quasi_im1_im2['pval'], method='fdr_bh')
    kruskal_TMS_real_quasi_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(kruskal_TMS_real_quasi_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/kruskal_TMS_real_quasi_im1_im2.xlsx", header=True, index=False)


data_files = ['DAY2_SHAM.mat']

data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}_background.xlsx"

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_real = df.loc[df['class'] == 'right_real']
    df_quasi = df.loc[df['class'] == 'right_quasi']
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    kruskal_sham_real_quasi_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_real = df_real[column]
            curr_quasi = df_quasi[column]
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            stat, pval = kruskal(curr_real, curr_im1, curr_im1, curr_im2)
            if not np.isnan(pval):
                kruskal_sham_real_quasi_im1_im2['pval'].append(pval)
                kruskal_sham_real_quasi_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_sham_real_quasi_im1_im2['pval'], method='fdr_bh')
    kruskal_sham_real_quasi_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(kruskal_sham_real_quasi_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/kruskal_sham_real_quasi_im1_im2.xlsx", header=True, index=False)


data_files = ['1st_Day_background_recordings.mat']

data_files_wo_extension = [data_file[:-4] for data_file in data_files]
xlsx_file = f"{dataframe_path}/dataframe_{'_'.join(data_files_wo_extension)}_background.xlsx"

if os.path.isfile(xlsx_file):
    df = pd.read_excel(xlsx_file)
    df_real = df.loc[df['class'] == 'right_real']
    df_quasi = df.loc[df['class'] == 'right_quasi']
    df_im1 = df.loc[df['class'] == 'right_im1']
    df_im2 = df.loc[df['class'] == 'right_im2']
    kruskal_1st_day_real_quasi_im1_im2 = {'feature': [], 'pval': [], 'pval_bh': []}
    for column in df:
        if column != 'class' and column != 'trial':
            curr_real = df_real[column]
            curr_quasi = df_quasi[column]
            curr_im1 = df_im1[column]
            curr_im2 = df_im2[column]
            stat, pval = kruskal(curr_real, curr_im1, curr_im1, curr_im2)
            if not np.isnan(pval):
                kruskal_1st_day_real_quasi_im1_im2['pval'].append(pval)
                kruskal_1st_day_real_quasi_im1_im2['feature'].append(column)
    reject, pval_bh, alphacSidak, alphacBonf = multipletests(kruskal_1st_day_real_quasi_im1_im2['pval'],
                                                             method='fdr_bh')
    kruskal_1st_day_real_quasi_im1_im2['pval_bh'] = pval_bh
    df = pd.DataFrame.from_dict(kruskal_1st_day_real_quasi_im1_im2)
    df.to_excel(f"{dataframe_stat_path}/kruskal_1st_day_real_quasi_im1_im2.xlsx", header=True, index=False)
