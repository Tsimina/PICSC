# # spotify_xgboost_simple_clean.py

# import os
# import warnings
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
# )
# import xgboost as xgb
# import matplotlib.pyplot as plt  # ✅ added

# warnings.filterwarnings("ignore")

# # ✅ Put your dataset path here (no terminal / no input)
# FILEPATH = r"E:\master an 2\picsc proiect\start code\extracted_features_all_flows.csv"

# RANDOM_STATE = 42
# TEST_SIZE = 0.10
# VAL_SIZE = 0.10


# def load_data(filepath: str):
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"CSV file not found: {filepath}")

#     df = pd.read_csv(filepath)

#     if "Label" not in df.columns:
#         raise ValueError("Expected a 'Label' column in the CSV.")

#     # Binary target: Spotify=1, Others=0
#     df["target"] = (df["Label"] == "Spotify").astype(int)

#     exclude_cols = {"Flow_ID", "Label", "Source_File", "target"}
#     feature_cols = [c for c in df.columns if c not in exclude_cols]

#     if not feature_cols:
#         raise ValueError("No feature columns found after excluding metadata columns.")

#     # Ensure features are numeric (coerce non-numeric to NaN)
#     df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

#     print(f"Loaded: {len(df)} rows | Features: {len(feature_cols)}")
#     pos = int(df["target"].sum())
#     neg = int(len(df) - pos)
#     print(f"Class distribution -> Spotify: {pos} | Others: {neg}")

#     return df, feature_cols


# def split_80_10_10(df: pd.DataFrame):
#     # First: 90% (train+val) / 10% test
#     train_val_df, test_df = train_test_split(
#         df,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=df["target"],
#     )

#     # Second: split train_val into train (80%) and val (10%)
#     val_relative = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.10 / 0.90
#     train_df, val_df = train_test_split(
#         train_val_df,
#         test_size=val_relative,
#         random_state=RANDOM_STATE,
#         stratify=train_val_df["target"],
#     )

#     return train_df, val_df, test_df


# def impute_with_train_means(train_df, val_df, test_df, feature_cols):
#     train_means = train_df[feature_cols].mean(numeric_only=True)

#     X_train = train_df[feature_cols].fillna(train_means)
#     y_train = train_df["target"].astype(int)

#     X_val = val_df[feature_cols].fillna(train_means)
#     y_val = val_df["target"].astype(int)

#     X_test = test_df[feature_cols].fillna(train_means)
#     y_test = test_df["target"].astype(int)

#     return X_train, y_train, X_val, y_val, X_test, y_test


# def train_model(X_train, y_train, X_val, y_val):
#     model = xgb.XGBClassifier(
#         objective="binary:logistic",
#         max_depth=6,
#         learning_rate=0.1,
#         n_estimators=1000,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=RANDOM_STATE,
#         n_jobs=-1,
#         eval_metric="logloss",
#         early_stopping_rounds=50,  # ✅ here (works with your xgboost)
#     )

#     model.fit(
#         X_train,
#         y_train,
#         eval_set=[(X_val, y_val)],
#         verbose=50,
#     )

#     best_iter = getattr(model, "best_iteration", None)
#     if best_iter is not None:
#         print(f"\nTraining finished. Best iteration: {best_iter}")
#     else:
#         print("\nTraining finished.")

#     return model


# def plot_and_save_confusion_matrix(cm, class_names, out_path, title):
#     """
#     Saves a confusion matrix as a PNG image.
#     """
#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(cm, cmap=plt.cm.Greens)

#     ax.set_title(title)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("Actual")

#     ax.set_xticks(range(len(class_names)))
#     ax.set_yticks(range(len(class_names)))
#     ax.set_xticklabels(class_names)
#     ax.set_yticklabels(class_names)

#     # Put values in cells
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, str(cm[i, j]), ha="center", va="center")

#     fig.tight_layout()
#     plt.show()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)


# def print_final_metrics(model, X, y, title="TEST SET"):
#     y_pred = model.predict(X)

#     acc = accuracy_score(y, y_pred)
#     prec = precision_score(y, y_pred, zero_division=0)
#     rec = recall_score(y, y_pred, zero_division=0)
#     f1 = f1_score(y, y_pred, zero_division=0)
#     cm = confusion_matrix(y, y_pred)

#     print("\n" + "=" * 60)
#     print(f"FINAL METRICS ({title})")
#     print("=" * 60)
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")
#     print(f"F1-score : {f1:.4f}")

#     print("\nConfusion Matrix (rows=actual, cols=pred):")
#     print(cm)

#     print("\nClassification Report:")
#     print(
#         classification_report(
#             y,
#             y_pred,
#             target_names=["Others", "Spotify"],
#             digits=4,
#             zero_division=0,
#         )
#     )

#     # ✅ Save confusion matrix image
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     out_path = os.path.join(script_dir, "confusion_matrix.png")

#     plot_and_save_confusion_matrix(
#         cm=cm,
#         class_names=["Others", "Spotify"],
#         out_path=out_path,
#         title=f"Confusion Matrix ({title})",
#     )

#     print(f"\n🖼️ Confusion matrix image saved to:\n{out_path}")


# def main():
#     print("=" * 60)
#     print("SPOTIFY vs OTHERS - XGBoost (80/10/10)")
#     print("=" * 60)

#     df, feature_cols = load_data(FILEPATH)
#     train_df, val_df, test_df = split_80_10_10(df)

#     print("\nSplit sizes:")
#     print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
#     print(f"Val  : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
#     print(f"Test : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

#     X_train, y_train, X_val, y_val, X_test, y_test = impute_with_train_means(
#         train_df, val_df, test_df, feature_cols
#     )

#     model = train_model(X_train, y_train, X_val, y_val)

#     # Display metrics at the end of training (on test set)
#     print_final_metrics(model, X_test, y_test, title="TEST SET")


# if __name__ == "__main__":
#     main()


# spotify_xgboost_simple_clean.py

import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ✅ Put your dataset path here (no terminal / no input)
FILEPATH = r"E:\master an 2\picsc proiect\start code\extracted_features_all_flows.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.10
VAL_SIZE = 0.10


def load_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the CSV.")

    # Binary target: Spotify=1, Others=0
    df["target"] = (df["Label"] == "Spotify").astype(int)

    exclude_cols = {"Flow_ID", "Label", "Source_File", "target"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    # Ensure features are numeric (coerce non-numeric to NaN)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Loaded: {len(df)} rows | Features: {len(feature_cols)}")
    pos = int(df["target"].sum())
    neg = int(len(df) - pos)
    print(f"Class distribution -> Spotify: {pos} | Others: {neg}")

    return df, feature_cols


def split_80_10_10(df: pd.DataFrame):
    # First: 90% (train+val) / 10% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    # Second: split train_val into train (80%) and val (10%)
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)  # 0.10 / 0.90
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        random_state=RANDOM_STATE,
        stratify=train_val_df["target"],
    )

    return train_df, val_df, test_df


def impute_with_train_means(train_df, val_df, test_df, feature_cols):
    train_means = train_df[feature_cols].mean(numeric_only=True)

    X_train = train_df[feature_cols].fillna(train_means)
    y_train = train_df["target"].astype(int)

    X_val = val_df[feature_cols].fillna(train_means)
    y_val = val_df["target"].astype(int)

    X_test = test_df[feature_cols].fillna(train_means)
    y_test = test_df["target"].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def grid_search_xgb(X_train, y_train, X_val, y_val, select_by="f1"):
    """
    Manual grid search on Train/Val.
    For each param set, evaluates multiple metrics on VAL:
    accuracy, precision, recall, f1.

    select_by: "f1" (default) or "accuracy" or "precision" or "recall"
    """
    # ✅ Keep this grid reasonably small; expand if you want
    param_grid = {
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
    }

    base_params = dict(
        objective="binary:logistic",
        n_estimators=2000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

    grid = list(ParameterGrid(param_grid))
    print("\n" + "=" * 70)
    print(f"GRID SEARCH: {len(grid)} combinations | select_by={select_by.upper()} (VAL)")
    print("=" * 70)

    if select_by not in {"f1", "accuracy", "precision", "recall"}:
        raise ValueError("select_by must be one of: f1, accuracy, precision, recall")

    results = []
    best_score = -1.0
    best_params = None
    best_model = None

    for i, p in enumerate(grid, start=1):
        params = {**base_params, **p}
        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(X_val)

        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)

        row = {
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            **p,
        }
        results.append(row)

        current_score = row[select_by]
        print(
            f"[{i:02d}/{len(grid):02d}] "
            f"ACC={val_acc:.4f} | P={val_prec:.4f} | R={val_rec:.4f} | F1={val_f1:.4f} "
            f"| params={p}"
        )

        if current_score > best_score:
            best_score = current_score
            best_params = p
            best_model = model

    # Show top 5 configs by F1 (handy to inspect)
    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print("TOP 5 PARAM SETS BY VAL F1:")
    print(results_df.head(5).to_string(index=False))
    print("-" * 70)

    print(f"\nBEST BY {select_by.upper()}: {best_score:.4f}")
    print(f"BEST PARAMS: {best_params}")
    best_iter = getattr(best_model, "best_iteration", None)
    if best_iter is not None:
        print(f"BEST ITERATION (early stop): {best_iter}")

    return best_model, best_params, best_score, results_df



def train_final_model_on_train_val(X_train, y_train, X_val, y_val, best_model, best_params):
    """
    Train final model on Train+Val using best hyperparameters.
    Uses the best_iteration from early stopping as n_estimators (if available).
    """
    X_train_val = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    best_iter = getattr(best_model, "best_iteration", None)
    final_n_estimators = (best_iter + 1) if best_iter is not None else 500

    final_params = dict(
        objective="binary:logistic",
        n_estimators=final_n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        **best_params,
    )

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train_val, y_train_val, verbose=False)

    print(f"\nFinal model trained on Train+Val with n_estimators={final_n_estimators}")
    return final_model


def plot_and_save_confusion_matrix(cm, class_names, out_path, title, cmap=plt.cm.Greens):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    plt.show()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_final_metrics(model, X, y, title="TEST SET"):
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print("\n" + "=" * 60)
    print(f"FINAL METRICS ({title})")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix (rows=actual, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=["Others", "Spotify"],
            digits=4,
            zero_division=0,
        )
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "confusion_matrix_grid_search.png")

    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=["Others", "Spotify"],
        out_path=out_path,
        title=f"Confusion Matrix ({title})",
        cmap=plt.cm.Greens,
    )

    print(f"\n🖼️ Confusion matrix image saved to:\n{out_path}")


def show_feature_importance(model, feature_cols, top_n=None):
    """
    Prints and plots XGBoost feature importance.
    Uses model.feature_importances_ (gain-like importance in sklearn wrapper).
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("\nNo feature_importances_ found on the model.")
        return

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        imp_df = imp_df.head(top_n)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (higher = more impact)")
    print("=" * 60)
    print(imp_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("SPOTIFY vs OTHERS - XGBoost (80/10/10)")
    print("=" * 60)

    df, feature_cols = load_data(FILEPATH)
    train_df, val_df, test_df = split_80_10_10(df)

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val  : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    X_train, y_train, X_val, y_val, X_test, y_test = impute_with_train_means(
        train_df, val_df, test_df, feature_cols
    )

    # ✅ Grid search on Train/Val
    best_model, best_params, best_score, gs_results = grid_search_xgb(
        X_train, y_train, X_val, y_val, select_by="f1"
)

    # ✅ Train final model on Train+Val using best params
    final_model = train_final_model_on_train_val(
        X_train, y_train, X_val, y_val, best_model, best_params
    )
    
    show_feature_importance(final_model, feature_cols, top_n=15)  # or top_n=None for all

    # ✅ Evaluate on Test
    print_final_metrics(final_model, X_test, y_test, title="TEST SET")


if __name__ == "__main__":
    main()
