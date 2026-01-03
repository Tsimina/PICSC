import numpy as np
import pandas as pd

# ---- Metrics ---- #
def code_one_hot(Y_int, Kclass=2):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(0, DB_size):
      Y_onehot[i,Y_int[i]] = 1
    return Y_onehot

def get_ua(OUT, TAR):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.absolute(aux))//2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN/VN)/Kclass*100, decimals=4)
    return UA

def get_wa(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits/DB_size*100, decimals=4)
    return WA
# ----------------- #

# ------ Data manipulation ------ #
def read_csv(file_path):
  # Citirea fișierului CSV
    data = pd.read_csv(file_path)
    data = data.loc[:, ~data.columns.isin(['Gen', 'Rec_Idx'])]
    return data

# Reading split information from file
def read_splits_txt(file_path): 
    splits = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:  # Citim split-urile
            train_part, val_part = line.strip().split(" Val: ")
            train_subjects = list(eval(train_part.replace("Train: ", "")))
            val_subjects = list(eval(val_part))
            splits.append((train_subjects, val_subjects))
    return splits


def retrieve_samples(data, split_idx):
    split_idx = map(int, split_idx)
    split_df = data[data['ID'].isin(split_idx)]

    X = split_df.iloc[:, 2:].to_numpy(dtype=np.float32)    # Features
    Y = split_df.iloc[:, 1].map({"lie": 0, "truth": 1})    # Classes 

    return np.array(X), np.array(Y) 
# -------------------------------------- #