# spotify_decision_tree_grouped_gridsearch.py

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ✅ Put your dataset path here (no terminal / no input)
FILEPATH = r"E:\master an 2\picsc proiect\start code\extracted_features_all_flows_15s.csv"
RANDOM_STATE = 42


def pick_group_column(df: pd.DataFrame) -> str:
    for c in ["Source_File", "Source_PCAP"]:
        if c in df.columns:
            return c
    raise ValueError("No group column found. Need 'Source_File' or 'Source_PCAP' for grouped CV.")


def load_data(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    df = pd.read_csv(filepath)

    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column in the CSV.")

    group_col = pick_group_column(df)

    # Binary target: Spotify=1, Others=0
    df["target"] = (df["Label"] == "Spotify").astype(int)

    exclude_cols = {"Flow_ID", "Label", "target", group_col}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata columns.")

    # Coerce features to numeric
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Loaded: {len(df)} rows | Features: {len(feature_cols)} | Group col: {group_col}")
    pos = int(df["target"].sum())
    neg = int(len(df) - pos)
    print(f"Class distribution -> Spotify: {pos} | Others: {neg}")
    print(f"Unique groups: {df[group_col].nunique()}")

    return df, feature_cols, group_col


def grouped_split_80_10_10(df: pd.DataFrame, group_col: str):
    """
    Group-safe split ~80/10/10 using StratifiedGroupKFold.
    - First: pick 1 fold out of 10 as TEST (~10%)
    - Second: from remaining 90%, pick 1 fold out of 9 as VAL (~10% of full)
    """
    X_dummy = df.index.values
    y = df["target"].values
    groups = df[group_col].values

    # 1) TEST
    sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    trval_idx, test_idx = next(sgkf_test.split(X_dummy, y, groups))

    train_val_df = df.iloc[trval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # 2) VAL from train_val
    X_dummy2 = train_val_df.index.values
    y2 = train_val_df["target"].values
    g2 = train_val_df[group_col].values

    sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sgkf_val.split(X_dummy2, y2, g2))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

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


def evaluate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def grid_search_dt(X_train, y_train, X_val, y_val, select_by="f1"):
    if select_by not in {"f1", "accuracy", "precision", "recall"}:
        raise ValueError("select_by must be one of: f1, accuracy, precision, recall")

    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10, 0.1],
        "min_samples_leaf": [1, 2, 4, 0.05],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }

    grid = list(ParameterGrid(param_grid))
    print("\n" + "=" * 70)
    print(f"DT GRID SEARCH: {len(grid)} combinations | select_by={select_by.upper()} (VAL)")
    print("=" * 70)

    best_score = -1.0
    best_params = None
    rows = []

    for i, p in enumerate(grid, start=1):
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, **p)
        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        m = evaluate_metrics(y_val, y_val_pred)
        rows.append({**p, **m})

        print(
            f"[{i:03d}/{len(grid):03d}] "
            f"ACC={m['accuracy']:.4f} | P={m['precision']:.4f} | R={m['recall']:.4f} | F1={m['f1']:.4f} "
            f"| params={p}"
        )

        if m[select_by] > best_score:
            best_score = m[select_by]
            best_params = p

    results_df = pd.DataFrame(rows).sort_values(select_by, ascending=False).reset_index(drop=True)

    print("\n" + "-" * 70)
    print(f"TOP 5 PARAM SETS BY VAL {select_by.upper()}:")
    print(results_df.head(5).to_string(index=False))
    print("-" * 70)

    print(f"\nBEST VAL {select_by.upper()}: {best_score:.4f}")
    print(f"BEST PARAMS: {best_params}")

    return best_params, best_score, results_df


def plot_and_save_confusion_matrix(cm, class_names, out_path, title, cmap="Greens"):
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


def show_feature_importance(model, feature_cols, top_n=15):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("\nNo feature_importances_ found.")
        return

    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Decision Tree)")
    print("=" * 60)
    print(imp_df.head(top_n).to_string(index=False))

    plot_df = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (DT)")
    plt.tight_layout()
    plt.show()


def plot_tree_and_save(model, feature_cols, out_path, max_depth=4):
    plt.figure(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=["Others", "Spotify"],
        filled=True,
        max_depth=max_depth,
        fontsize=8,
    )
    plt.title(f"Decision Tree (shown up to depth={max_depth})")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_final_metrics_and_cm(model, X_test, y_test, title="GROUPED TEST SET"):
    y_pred = model.predict(X_test)
    m = evaluate_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(f"FINAL METRICS ({title})")
    print("=" * 60)
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1-score : {m['f1']:.4f}")

    print("\nConfusion Matrix (rows=actual, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Others", "Spotify"],
            digits=4,
            zero_division=0,
        )
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "dt_grouped_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        cm,
        class_names=["Others", "Spotify"],
        out_path=out_path,
        title="Decision Tree Confusion Matrix (GROUPED TEST SET)",
        cmap="Greens",
    )
    print(f"\n🖼️ Confusion matrix image saved to:\n{out_path}")


def main():
    print("=" * 60)
    print("SPOTIFY vs OTHERS - Decision Tree (GROUPED ~80/10/10 + Grid Search)")
    print("=" * 60)

    df, feature_cols, group_col = load_data(FILEPATH)

    train_df, val_df, test_df = grouped_split_80_10_10(df, group_col)

    print("\nSplit sizes (grouped, approx):")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val  : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test : {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # sanity check: no group leakage
    assert set(train_df[group_col]).isdisjoint(set(val_df[group_col]))
    assert set(train_df[group_col]).isdisjoint(set(test_df[group_col]))
    assert set(val_df[group_col]).isdisjoint(set(test_df[group_col]))

    X_train, y_train, X_val, y_val, X_test, y_test = impute_with_train_means(
        train_df, val_df, test_df, feature_cols
    )

    best_params, best_score, gs_results = grid_search_dt(
        X_train, y_train, X_val, y_val, select_by="f1"
    )

    X_train_val = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    final_model = DecisionTreeClassifier(random_state=RANDOM_STATE, **best_params)
    final_model.fit(X_train_val, y_train_val)

    show_feature_importance(final_model, feature_cols, top_n=15)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tree_path = os.path.join(script_dir, "decision_tree_plot.png")
    plot_tree_and_save(final_model, feature_cols, tree_path, max_depth=4)
    print(f"\n🌳 Decision tree plot saved to:\n{tree_path}")

    print_final_metrics_and_cm(final_model, X_test, y_test, title="GROUPED TEST SET")


if __name__ == "__main__":
    main()
