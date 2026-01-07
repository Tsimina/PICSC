import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
from time import time
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid display issues
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
    # Load dataset for Spotify vs Rest
    dataset_path = '../extracted_features/dataset_timeout600_flow60_balanced.csv'
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

    # Hyperparameters
    estimators = list(range(5, 55, 5))
    min_samples_split = [2, 3, 5]
    max_depth = list(range(5, 15))
    min_samples_leaf = [1, 3]
    max_samples = [0.7, 0.9]
    max_features = ['sqrt', 'log2']

    # Compute no simulations to perform
    Nsim = len(estimators)*len(min_samples_split)*len(max_depth)*len(min_samples_leaf)*len(max_samples)*len(max_features)

    idx_sim = 0
    METRIX_ = np.zeros((Nsim, 4))  # train_acc, val_acc, train_f1, val_f1
  

    for es in estimators:
      for mss in min_samples_split:
        for md in max_depth:
          for msl in min_samples_leaf:
            for ms in max_samples:
              for mf in max_features:
                train_accuracies = []
                val_accuracies = []
                train_f1s = []
                val_f1s = []
                
                for split_index, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                  
                  X_train_split, Y_train = X[train_idx], y[train_idx]
                  X_val_split, Y_val = X[val_idx], y[val_idx]

                  # Create the model
                  MODEL = RF(n_estimators=es, min_samples_split=mss, max_depth=md,
                      min_samples_leaf=msl, max_samples=ms, max_features=mf)
                  start = time()
                  MODEL.fit(X_train_split, Y_train)
                  end = time()

                  print('Training time: %.2f sec' % (end-start))
                  y_train_pred = MODEL.predict(X_train_split)
                  y_val_pred = MODEL.predict(X_val_split)
                  
                  # Metrics
                  train_acc = accuracy_score(Y_train, y_train_pred)
                  val_acc = accuracy_score(Y_val, y_val_pred)
                  train_f1 = f1_score(Y_train, y_train_pred)
                  val_f1 = f1_score(Y_val, y_val_pred)
                  
                  train_accuracies.append(train_acc)
                  val_accuracies.append(val_acc)
                  train_f1s.append(train_f1)
                  val_f1s.append(val_f1)
                  
                  print(f'Fold {split_index+1}: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}, Train F1 = {train_f1:.3f}, Val F1 = {val_f1:.3f}')

                # Average over folds
                train_acc_avg = np.mean(train_accuracies) * 100
                val_acc_avg = np.mean(val_accuracies) * 100
                train_f1_avg = np.mean(train_f1s) * 100
                val_f1_avg = np.mean(val_f1s) * 100
                
                print(f'Estimators: {es}, Min Samples Split: {mss}, Max Depth: {md}, Min Samples Leaf: {msl}, Max Samples: {ms}, Max Features: {mf}')
                print(f'Avg Train Acc: {train_acc_avg:.2f}%, Avg Val Acc: {val_acc_avg:.2f}%, Avg Train F1: {train_f1_avg:.2f}%, Avg Val F1: {val_f1_avg:.2f}%\n')
                
                METRIX_[idx_sim, :] = [train_acc_avg, val_acc_avg, train_f1_avg, val_f1_avg]

                # Update best model if current is better
                if val_acc_avg > best_val_score:
                    best_val_score = val_acc_avg
                    best_model = MODEL
                    best_es = es
                    best_mss = mss
                    best_md = md
                    best_msl = msl
                    best_ms = ms
                    best_mf = mf
                    print(f"New best model: Estimators {es}, Min Samples Split {mss}, Max Depth {md}, Min Samples Leaf {msl}, Max Samples {ms}, Max Features {mf}, Val Acc {best_val_score:.2f}%")

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
                                      'Train_Acc [%]', 'Val_Acc [%]',
                                      'Train_F1 [%]', 'Val_F1 [%]'],
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
    results_path = os.path.join(results_dir, f'RF_binary_spotify_results_timeout600_flow60_balanced.csv')
    # Verifică dacă fișierul există
    if os.path.exists(results_path):
        # Scrie în fișier folosind append (fără header)
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        # Scrie în fișier cu header (pentru prima scriere)
        df.to_csv(results_path, index=False)
  
    print("\n=== Best Model Summary ===")
    print(f"Best Hyperparameters: Estimators={best_es}, Min Samples Split={best_mss}, Max Depth={best_md}, Min Samples Leaf={best_msl}, Max Samples={best_ms}, Max Features={best_mf}")
    print(f"Best Validation Accuracy: {best_val_score:.2f}%")
    
    # Print best configuration details
    best_idx = np.argmax(METRIX_[:, 1])  # index of max val_acc
    print(f"\nBest Configuration Details:")
    print(f"SIM: {sim_list_idx[best_idx]}")
    print(f"Estimators: {sim_list_estimators[best_idx]}")
    print(f"Min Samples Split: {sim_list_min_samples_split[best_idx]}")
    print(f"Max Depth: {sim_list_max_depth[best_idx]}")
    print(f"Min Samples Leaf: {sim_list_min_samples_leaf[best_idx]}")
    print(f"Max Samples: {sim_list_max_samples[best_idx]}")
    print(f"Max Features: {sim_list_max_features[best_idx]}")
    print(f"Train Acc: {METRIX_[best_idx, 0]:.2f}%, Val Acc: {METRIX_[best_idx, 1]:.2f}%")
    print(f"Train F1: {METRIX_[best_idx, 2]:.2f}%, Val F1: {METRIX_[best_idx, 3]:.2f}%")
    
    # Retrain best model on full data and compute confusion matrix
    best_model_full = RF(n_estimators=best_es, min_samples_split=best_mss,
                         max_depth=best_md, min_samples_leaf=best_msl,
                         max_samples=best_ms, max_features=best_mf)
    best_model_full.fit(X, y)
    y_pred_full = best_model_full.predict(X)
    
    # Save best model
    import joblib
    model_path = os.path.join(results_dir, 'best_rf_model_300.pth')
    joblib.dump(best_model_full, model_path)
    print(f"Best model saved to {model_path}")
    
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
    plot_path = os.path.join(results_dir, 'rf_confusion_matrix_timeout600_flow60_balanced.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.close()