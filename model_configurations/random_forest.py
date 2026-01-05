import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
from time import time
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt



def codeOneHot(Y_int, Kclass=2):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(0, DB_size):
      Y_onehot[i,Y_int[i]] = 1
    return Y_onehot

def getUA(OUT, TAR):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.absolute(aux))//2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN/VN)/Kclass*100, decimals=1)
    return UA

def getWA(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits/DB_size*100, decimals=1)
    return WA



if __name__ == '__main__':
  # Datasets
  root_path = "database/features"
  windows = [25] # Windows
  Kclass = 2

  # Hyperparameters
  estimators = list(range(5, 55, 5))
  min_samples_split = [ 0.1, 0.15, 0.2]
  max_depth = list(range(2, 15))
  min_samples_leaf = [0.01, 0.05, 0.1]
  max_samples = [0.1, 0.3, 0.5]
  max_features = ['sqrt', 'log2']

  # Compute no simulations to perform
  Nsim = len(estimators)*len(min_samples_split)*len(max_depth)*len(min_samples_leaf)*len(max_samples)*len(max_features)

  for window in windows:
    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))

    # Load dataset for Spotify vs Rest
    dataset_path = '../src/dataset_pcap.csv'
    df = pd.read_csv(dataset_path)
    
    # Features: exclude Flow_ID, Source_PCAP, Label
    feature_cols = [col for col in df.columns if col not in ['Flow_ID', 'Source_PCAP', 'Label']]
    X = df[feature_cols].values
    y = df['Label'].values  # 1 for Spotify, 0 for others
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    print(f"Dataset shape after imputation: {X.shape}, Labels: {np.unique(y, return_counts=True)}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Save best model for the current split
    best_model = None
    best_val_score = -float('inf')
    best_es = None
    best_mss = None
    best_md = None
    best_msl = None
    best_ms = None
    best_mf = None
  

    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                METRIX = []
                for split_index, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                  
                  X_train_split, Y_train = X[train_idx], y[train_idx]
                  X_val_split, Y_val = X[val_idx], y[val_idx]

                  # Normalize
                  scaler = StandardScaler()
                  X_train_scaled = scaler.fit_transform(X_train_split)
                  X_val_scaled = scaler.transform(X_val_split)

                  # Create the model
                  MODEL = RF(n_estimators=es, min_samples_split=mss, max_depth=md,
                      min_samples_leaf=msl, max_samples=ms, max_features=mf)
                  start = time()
                  MODEL.fit(X_train_scaled, Y_train)
                  end = time()

                  print('Training time: %.2f sec' % (end-start))
                  OUT_train = MODEL.predict(X_train_scaled)
                  OUT_val = MODEL.predict(X_val_scaled)
                  # Train metrics
                  UA_train = getUA(codeOneHot(OUT_train),
                                  codeOneHot(Y_train))
                  WA_train = getWA(codeOneHot(OUT_train),
                                  codeOneHot(Y_train))
                  print(f'UA (train) = {UA_train}. WA (train) = {WA_train}')
                  # Val metrics
                  UA_val = getUA(codeOneHot(OUT_val), codeOneHot(Y_val))
                  WA_val = getWA(codeOneHot(OUT_val), codeOneHot(Y_val))
                  print(f'UA (val) = {UA_val}. WA (val) = {WA_val}\n')
                  METRIX += [UA_train, WA_train, UA_val, WA_val]

                # -> Cross-validation results:
                UA_train_avg = WA_train_avg = UA_val_avg = WA_val_avg = 0
                L = len(METRIX)
                for i in range(0, L, 4):
                    UA_train_avg += METRIX[i]
                UA_train_avg = np.round(UA_train_avg/5, decimals=2)
                for i in range(1, L, 4):
                    WA_train_avg += METRIX[i]
                WA_train_avg = np.round(WA_train_avg/5, decimals=2)
                for i in range(2, L, 4):
                    UA_val_avg += METRIX[i]
                UA_val_avg = np.round(UA_val_avg/5, decimals=2)
                for i in range(3, L, 4):
                    WA_val_avg += METRIX[i]
                WA_val_avg = np.round(WA_val_avg/5, decimals=2)
                print(f'UA avg (train) = {UA_train_avg}. WA avg (train) = {WA_train_avg}')
                print(f'UA avg (val) = {UA_val_avg}. WA avg (val) = {WA_val_avg}\n')
                METRIX_[idx_sim,:] = [UA_train_avg, WA_train_avg,
                                        UA_val_avg, WA_val_avg]

                # Update best model if current is better
                if UA_val_avg > best_val_score:
                    best_val_score = UA_val_avg
                    best_model = MODEL
                    best_es = es
                    best_mss = mss
                    best_md = md
                    best_msl = msl
                    best_ms = ms
                    best_mf = mf
                    print(f"New best model found with: {es} estimators, {mss} minimum sample split",
                      f"{md} max depth, {msl} minimum sample per leaf, {ms} max sample count/percentage, "
                      f"{mf} max features. Val Mean UA: {best_val_score:.2f}")

                idx_sim += 1
                if idx_sim % 100 == 0:
                    print(f"Completed {idx_sim}/{Nsim} simulations")

      sim_list_idx = range(0, Nsim)
      sim_list_estimators = []
      sim_list_min_samples_split = []
      sim_list_max_depth = []
      sim_list_min_samples_leaf = []
      sim_list_max_samples = []
      sim_list_max_features = []
      for es in estimators:
        for mss in min_samples_split:
          for md in max_depth:
            for msl in min_samples_leaf:
              for ms in max_samples:
                for mf in max_features:
                  sim_list_estimators.append(es)
                  sim_list_min_samples_split.append(mss)
                  sim_list_max_depth.append(md)
                  sim_list_min_samples_leaf.append(msl)
                  sim_list_max_samples.append(ms)
                  sim_list_max_features.append(mf)

      # Save best model
      rf_path = os.getcwd()
      os.makedirs(rf_path, exist_ok=True)

      df_dict = { k:v for (k, v) in zip(['SIM', 'Es', 'Mss', 'Md', 'Msl', 'Ms', 'Mf',
                                        'UA_train [%]', 'WA_train [%]',
                                        'UA_val [%]', 'WA_val [%]'],
                                        [sim_list_idx,
                                        sim_list_estimators,
                                        sim_list_min_samples_split,
                                        sim_list_max_depth,
                                        sim_list_min_samples_leaf,
                                        sim_list_max_samples,
                                        sim_list_max_features,
                                        METRIX_[:,0], METRIX_[:,1],
                                        METRIX_[:,2], METRIX_[:,3]]) }
      df = pd.DataFrame(df_dict)
      results_dir = '../results'
      os.makedirs(results_dir, exist_ok=True)
      results_path = os.path.join(results_dir, f'RF_binary_spotify_results.csv')
      # Verifică dacă fișierul există
      if os.path.exists(results_path):
          # Scrie în fișier folosind append (fără header)
          df.to_csv(results_path, mode='a', header=False, index=False)
      else:
          # Scrie în fișier cu header (pentru prima scriere)
          df.to_csv(results_path, index=False)
    
    # Save best model
    import joblib
    model_path = os.path.join(results_dir, 'best_rf_model.pth')
    joblib.dump(best_model_full, model_path)
    print(f"Best model saved to {model_path}")
    
    print("\n=== Best Model Summary ===")
    print(f"Best Hyperparameters: Estimators={best_es}, Min Samples Split={best_mss}, Max Depth={best_md}, Min Samples Leaf={best_msl}, Max Samples={best_ms}, Max Features={best_mf}")
    print(f"Best Validation Accuracy: {best_val_score:.2f}%")
    
    # Print best configuration details
    best_idx = np.argmax(METRIX_[:, 2])  # index of max UA_val
    print(f"\nBest Configuration Details:")
    print(f"SIM: {sim_list_idx[best_idx]}")
    print(f"Estimators: {sim_list_estimators[best_idx]}")
    print(f"Min Samples Split: {sim_list_min_samples_split[best_idx]}")
    print(f"Max Depth: {sim_list_max_depth[best_idx]}")
    print(f"Min Samples Leaf: {sim_list_min_samples_leaf[best_idx]}")
    print(f"Max Samples: {sim_list_max_samples[best_idx]}")
    print(f"Max Features: {sim_list_max_features[best_idx]}")
    print(f"Train UA: {METRIX_[best_idx, 0]:.2f}%, Train WA: {METRIX_[best_idx, 1]:.2f}%")
    print(f"Val UA: {METRIX_[best_idx, 2]:.2f}%, Val WA: {METRIX_[best_idx, 3]:.2f}%")
    
    # Retrain best model on full data and compute confusion matrix
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    
    best_model_full = RF(n_estimators=best_es, min_samples_split=best_mss,
                         max_depth=best_md, min_samples_leaf=best_msl,
                         max_samples=best_ms, max_features=best_mf)
    best_model_full.fit(X_scaled, y)
    y_pred_full = best_model_full.predict(X_scaled)
    
    cm = confusion_matrix(y, y_pred_full)
    print("\nClassification Report:")
    print(classification_report(y, y_pred_full, target_names=['Rest', 'Spotify']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Random Forest (Spotify vs Rest)')
    plt.colorbar()
    classes = ['Rest', 'Spotify']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    # Save the plot
    plot_path = os.path.join(results_dir, 'rf_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.close()