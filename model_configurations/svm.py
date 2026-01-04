import numpy as np
import pandas as pd
from time import time
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Binary classification: Spotify (1) vs Rest (0)
if __name__ == '__main__':
    # Load dataset
    dataset_path = '../src/dataset.csv'
    df = pd.read_csv(dataset_path)
    
    # Features: exclude Flow_ID, Source_PCAP, Label
    feature_cols = [col for col in df.columns if col not in ['Flow_ID', 'Source_PCAP', 'Label']]
    X = df[feature_cols].values
    y = df['Label'].values  # 1 for Spotify, 0 for others
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    print(f"Dataset shape after imputation: {X.shape}, Labels: {np.unique(y, return_counts=True)}")
    
    # Hyperparameters
    PCA_components = range(28, 1, -1)
    SVM_kernels = ['rbf', 'sigmoid', 'linear', 'poly']
    Cs = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    
    # Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    Nsim = len(PCA_components) * len(SVM_kernels) * len(Cs)
    METRIX_ = np.zeros((Nsim, 4))  # UA_train, WA_train, UA_val, WA_val (for binary, UA=WA=accuracy)
    
    best_model = None
    best_val_score = -float('inf')
    best_pca = None
    best_kernel = None
    best_C = None
    idx_sim = 0
    
    for pca_comp in PCA_components:
        for SVM_kernel in SVM_kernels:
            for C in Cs:
                train_accuracies = []
                val_accuracies = []
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Normalize
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # PCA
                    if pca_comp < X_train_scaled.shape[1]:
                        pca = PCA(n_components=pca_comp)
                        X_train_pca = pca.fit_transform(X_train_scaled)
                        X_val_pca = pca.transform(X_val_scaled)
                    else:
                        X_train_pca = X_train_scaled
                        X_val_pca = X_val_scaled
                    
                    # SVM
                    model = SVC(C=C, kernel=SVM_kernel)
                    start = time()
                    model.fit(X_train_pca, y_train)
                    end = time()
                    print(f'Training time for fold {fold+1}: {end-start:.2f} sec')
                    
                    # Predictions
                    y_train_pred = model.predict(X_train_pca)
                    y_val_pred = model.predict(X_val_pca)
                    
                    # Metrics (for binary, accuracy)
                    train_acc = accuracy_score(y_train, y_train_pred)
                    val_acc = accuracy_score(y_val, y_val_pred)
                    
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)
                    
                    print(f'Fold {fold+1}: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}')
                
                # Average over folds
                UA_train_avg = np.mean(train_accuracies) * 100
                WA_train_avg = UA_train_avg  # Same for binary
                UA_val_avg = np.mean(val_accuracies) * 100
                WA_val_avg = UA_val_avg
                
                print(f'PCA: {pca_comp}, Kernel: {SVM_kernel}, C: {C}')
                print(f'Avg Train Acc: {UA_train_avg:.2f}%, Avg Val Acc: {UA_val_avg:.2f}%\n')
                
                METRIX_[idx_sim, :] = [UA_train_avg, WA_train_avg, UA_val_avg, WA_val_avg]
                
                # Update best model
                if UA_val_avg > best_val_score:
                    best_val_score = UA_val_avg
                    best_model = model
                    best_pca = pca_comp
                    best_kernel = SVM_kernel
                    best_C = C
                    print(f"New best model: PCA {pca_comp}, Kernel {SVM_kernel}, C {C}, Val Acc {best_val_score:.2f}%")
                
                idx_sim += 1
    
    # Save results
    sim_list_idx = range(Nsim)
    sim_list_pca = []
    sim_list_kernels = []
    sim_list_Cs = []
    for pca_comp in PCA_components:
        for kernel in SVM_kernels:
            for C in Cs:
                sim_list_pca.append(pca_comp)
                sim_list_kernels.append(kernel)
                sim_list_Cs.append(C)
    
    df_results = pd.DataFrame({
        'SIM': sim_list_idx,
        'PCA_comp': sim_list_pca,
        'Kernel': sim_list_kernels,
        'C': sim_list_Cs,
        'UA_train [%]': METRIX_[:, 0],
        'WA_train [%]': METRIX_[:, 1],
        'UA_val [%]': METRIX_[:, 2],
        'WA_val [%]': METRIX_[:, 3]
    })
    
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'SVM_binary_spotify_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Save best model
    import joblib
    model_path = os.path.join(results_dir, 'best_svm_model.pth')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")
    
    print("\n=== Best Model Summary ===")
    print(f"Best Hyperparameters: PCA={best_pca}, Kernel={best_kernel}, C={best_C}")
    print(f"Best Validation Accuracy: {best_val_score:.2f}%")
    
    # Print best configuration details
    best_idx = np.argmax(METRIX_[:, 2])  # index of max UA_val
    print(f"\nBest Configuration Details:")
    print(f"SIM: {sim_list_idx[best_idx]}")
    print(f"PCA Components: {sim_list_pca[best_idx]}")
    print(f"Kernel: {sim_list_kernels[best_idx]}")
    print(f"C: {sim_list_Cs[best_idx]}")
    print(f"Train UA: {METRIX_[best_idx, 0]:.2f}%, Train WA: {METRIX_[best_idx, 1]:.2f}%")
    print(f"Val UA: {METRIX_[best_idx, 2]:.2f}%, Val WA: {METRIX_[best_idx, 3]:.2f}%")
    
    # Retrain best model on full data and compute confusion matrix
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    
    if best_pca is not None and best_pca < X_scaled.shape[1]:
        pca_full = PCA(n_components=best_pca)
        X_pca = pca_full.fit_transform(X_scaled)
    else:
        X_pca = X_scaled
    
    best_model_full = SVC(C=best_C, kernel=best_kernel)
    best_model_full.fit(X_pca, y)
    y_pred_full = best_model_full.predict(X_pca)
    
    cm = confusion_matrix(y, y_pred_full)
    print("\nClassification Report:")
    print(classification_report(y, y_pred_full, target_names=['Rest', 'Spotify']))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for SVM (Spotify vs Rest)')
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
    plot_path = os.path.join(results_dir, 'svm_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.close()