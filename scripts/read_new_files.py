import pickle
import mne

data_path = 'E:/YandexDisk/EEG/raw/MI_NN/29_chanels/Real/AO_0.1/'

with open(f'{data_path}/clf.pkl', "rb") as f:
    clf = pickle.load(f)

with open(f'{data_path}/clf_res.pkl', "rb") as f:
    clf_res = pickle.load(f)

with open(f'{data_path}/data.pkl', "rb") as f:
    data = pickle.load(f)

with open(f'{data_path}/data_full.pkl', "rb") as f:
    data_full = pickle.load(f)

with open(f'{data_path}/states_full.pkl', "rb") as f:
    states_full = pickle.load(f)

mne_python_raw = mne.io.read_raw_fif(f'{data_path}/mne_python_raw.fif')
o=0
