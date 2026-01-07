import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FILEPATH = '../extracted_features/dataset_timeout600_flow60.csv'
RANDOM_STATE = 42


def pick_group_column(df: pd.DataFrame) -> str:
    candidates = ["Source_File", "Source_PCAP", "pcap", "pcap_file", "Pcap_File"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a group column. Expected one of: "
        f"{candidates}. Add one (e.g., Source_File=pcap name) to avoid leakage."
    )


def coerce_numeric_dataframe(df: pd.DataFrame, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def make_xgb_classifier(scale_pos_weight=1.0, random_state=42):
    base_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        base_score=0.5,  # ensure valid base_score when a train fold has only one class
    )
    try:
        model = XGBClassifier(**base_params, device="cuda")
        _ = model.get_params()
        return model
    except Exception:
        return XGBClassifier(**base_params)


def plot_confusion_matrix(cm, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
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
    fig.savefig(out_path, dpi=200)
    plt.show()
    plt.close(fig)


def main():
    print("=" * 70)
    print("SPOTIFY vs REST — XGBoost + StratifiedGroupKFold + GridSearchCV")
    print("=" * 70)

    df = pd.read_csv(FILEPATH)

    if "Label" not in df.columns:
        raise ValueError("Expected a 'Label' column (Spotify vs Rest).")

    group_col = pick_group_column(df)
    # support both numeric labels (1 = Spotify, 0 = Rest) and string labels ("Spotify")
    lbl = df["Label"]
    if lbl.dtype == object or lbl.dtype.name == "category":
        df["target"] = lbl.astype(str).str.strip().str.lower().eq("spotify").astype(int)
    else:
        df["target"] = (pd.to_numeric(lbl, errors="coerce").fillna(0).astype(int) == 1).astype(int)

    drop_cols = {"Flow_ID", "Label", "target", group_col}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")

    df = coerce_numeric_dataframe(df, feature_cols)

    X = df[feature_cols]
    y = df["target"].astype(int).values
    groups = df[group_col].values

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw_all = (neg / max(pos, 1)) if pos > 0 else 1.0
    print(f"Rows={len(df)} | Features={len(feature_cols)} | Groups(col={group_col})={pd.Series(groups).nunique()}")
    print(f"Class counts: Spotify(pos)= {pos}, Rest(neg)= {neg}, scale_pos_weight= {spw_all:.3f}")

    preprocess = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    base_model = make_xgb_classifier(scale_pos_weight=spw_all, random_state=RANDOM_STATE)
    pipe_for_gs = Pipeline(steps=[("preprocess", preprocess), ("model", base_model)])

    param_grid = {
        "model__max_depth":        [4, 6, 8],
        "model__learning_rate":    [0.10, 0.05],
        "model__n_estimators":     [400, 800, 1200],
        "model__subsample":        [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }

    cv_gs = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    print("\nStarting GridSearchCV (StratifiedGroupKFold)...")
    grid_search = GridSearchCV(
        estimator=pipe_for_gs,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_gs,
        n_jobs=1,
        verbose=2,
        refit=True,
    )

    grid_search.fit(X, y, groups=groups)

    print("\n=== GRID SEARCH SUMMARY ===")
    print(f"Best mean CV AUC: {grid_search.best_score_:.4f}")
    print("Best params:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cols_to_show = [
        "mean_test_score", "std_test_score",
        "param_model__max_depth",
        "param_model__learning_rate",
        "param_model__n_estimators",
        "param_model__subsample",
        "param_model__colsample_bytree",
    ]
    print("\nTop 5 parameter sets by mean AUC:")
    print(
        cv_results[cols_to_show]
        .sort_values("mean_test_score", ascending=False)
        .head(5)
        .to_string(index=False)
    )

    best_model_in_pipe = grid_search.best_estimator_.named_steps["model"]
    best_xgb_params = {
        "n_estimators": best_model_in_pipe.get_params()["n_estimators"],
        "learning_rate": best_model_in_pipe.get_params()["learning_rate"],
        "max_depth": best_model_in_pipe.get_params()["max_depth"],
        "subsample": best_model_in_pipe.get_params()["subsample"],
        "colsample_bytree": best_model_in_pipe.get_params()["colsample_bytree"],
        "min_child_weight": best_model_in_pipe.get_params()["min_child_weight"],
        "reg_lambda": best_model_in_pipe.get_params()["reg_lambda"],
    }

    print("\nBest XGBoost params (clean):")
    for k, v in best_xgb_params.items():
        print(f"  {k}: {v}")

    print("\nRunning 5-fold StratifiedGroupKFold with tuned params (OOF predictions)...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_proba = np.zeros(len(X), dtype=float)
    oof_pred = np.zeros(len(X), dtype=int)
    fold_rows = []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        pos_tr = int((y_tr == 1).sum())
        neg_tr = int((y_tr == 0).sum())
        spw_fold = (neg_tr / max(pos_tr, 1)) if pos_tr > 0 else 1.0

        model_fold = make_xgb_classifier(scale_pos_weight=spw_fold, random_state=RANDOM_STATE + fold)
        model_fold.set_params(**best_xgb_params)

        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model_fold)])
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_proba[va] = proba
        oof_pred[va] = pred

        tn, fp, fn, tp = confusion_matrix(y_va, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        auc_val = roc_auc_score(y_va, proba) if len(np.unique(y_va)) == 2 else np.nan

        fold_rows.append({
            "fold": fold,
            "AUC": auc_val,
            "Accuracy": accuracy_score(y_va, pred),
            "Precision": precision_score(y_va, pred, zero_division=0),
            "Recall": recall_score(y_va, pred, zero_division=0),
            "F1": f1_score(y_va, pred, zero_division=0),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Pos(val)": int((y_va == 1).sum()),
            "Neg(val)": int((y_va == 0).sum()),
        })

    cm_all = confusion_matrix(y, oof_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_all.ravel()
    overall_sens = tp / (tp + fn) if (tp + fn) else 0.0
    overall_spec = tn / (tn + fp) if (tn + fp) else 0.0

    overall_auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) == 2 else np.nan
    overall = {
        "AUC": overall_auc,
        "Accuracy": accuracy_score(y, oof_pred),
        "Precision": precision_score(y, oof_pred, zero_division=0),
        "Recall": recall_score(y, oof_pred, zero_division=0),
        "F1": f1_score(y, oof_pred, zero_division=0),
    }

    print("\n=== 5-FOLD GROUPED CV (OOF) RESULTS ===")
    for r in fold_rows:
        print(
            f"Fold {r['fold']}: AUC={r['AUC']:.3f} | Acc={r['Accuracy']:.3f} | "
            f"P={r['Precision']:.3f} | R/Sens={r['Sensitivity']:.3f} | "
            f"Spec={r['Specificity']:.3f} | F1={r['F1']:.3f} "
            f"(Pos={r['Pos(val)']}, Neg={r['Neg(val)']})"
        )

    print("\nOVERALL OOF METRICS (grouped by PCAP/source):")
    print(
        f"AUC={overall['AUC']:.4f} | Accuracy={overall['Accuracy']:.4f} | "
        f"Precision={overall['Precision']:.4f} | Recall/Sens={overall_sens:.4f} | "
        f"Specificity={overall_spec:.4f} | F1={overall['F1']:.4f}"
    )

    print("\nClassification report (OOF):")
    unique_classes = np.unique(y)
    if len(unique_classes) == 2:
        labels = [0, 1]
        target_names = ["Rest", "Spotify"]
    else:
        single = int(unique_classes[0])
        labels = [single]
        target_names = ["Spotify"] if single == 1 else ["Rest"]
    print(classification_report(y, oof_pred, labels=labels, target_names=target_names, digits=4, zero_division=0))

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cm_path = os.path.join(script_dir, "xgb_confusion_matrix_binary_spotify_results_timeout600_flow60_streaming.png")
    plot_confusion_matrix(
        cm_all,
        class_names=["Rest", "Spotify"],
        title="XGBoost Grouped CV Confusion Matrix (OOF)",
        out_path=cm_path,
    )
    print(f"\nConfusion matrix saved to: {cm_path}")

    # ROC plot only when both classes are present
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, oof_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"OOF ROC (AUC={overall['AUC']:.3f})", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("XGBoost — Grouped CV ROC (OOF)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(script_dir, "xgb_roc_binary_spotify_results_timeout600_flow60.png")
        plt.savefig(roc_path, dpi=200)
        plt.show()
    else:
        roc_path = None
        print("Skipping ROC plot: only one class present in true labels.")

    print("\nRefitting final model on ALL data for feature importance...")

    final_model = make_xgb_classifier(scale_pos_weight=spw_all, random_state=2025)
    final_model.set_params(**best_xgb_params)

    final_pipe = Pipeline(steps=[("preprocess", preprocess), ("model", final_model)])
    final_pipe.fit(X, y)

    importances = final_pipe.named_steps["model"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    top_n = min(25, len(imp_df))
    print("\nTop feature importances:")
    print(imp_df.head(top_n).to_string(index=False))

    plt.figure(figsize=(8, max(4, 0.35 * top_n)))
    plot_df = imp_df.head(top_n).iloc[::-1]
    plt.barh(plot_df["feature"], plot_df["importance"], color="blue")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances — XGBoost (Grouped CV tuned)")
    plt.tight_layout()
    plt.show()

    try:
        cv_results_out = os.path.join(script_dir, "xgb_grid_search_cv_results.csv")
        cv_results.to_csv(cv_results_out, index=False)

        folds_out = os.path.join(script_dir, "xgb_groupcv_folds.csv")
        pd.DataFrame(fold_rows).to_csv(folds_out, index=False)

        overall_out = os.path.join(script_dir, "xgb_groupcv_overall.csv")
        overall_record = {
            "AUC": overall["AUC"],
            "Accuracy": overall["Accuracy"],
            "Precision": overall["Precision"],
            "Recall": overall["Recall"],
            "F1": overall["F1"],
            "Sensitivity": overall_sens,
            "Specificity": overall_spec,
            "Pos(total)": int((y == 1).sum()),
            "Neg(total)": int((y == 0).sum()),
        }
        pd.DataFrame([overall_record]).to_csv(overall_out, index=False)

        imp_out = os.path.join(script_dir, "xgb_feature_importances.csv")
        imp_df.to_csv(imp_out, index=False)

        cm_out = os.path.join(script_dir, "xgb_confusion_matrix.csv")
        pd.DataFrame(cm_all, index=["Rest", "Spotify"], columns=["Rest", "Spotify"]).to_csv(cm_out)

        oof_out = os.path.join(script_dir, "XGB_oof_predictions_binary_spotify_results_timeout600_flow60_balanced_streaming.csv")
        pd.DataFrame({
            group_col: groups,
            "true": y,
            "pred": oof_pred,
            "proba": oof_proba
        }).to_csv(oof_out, index=False)

        print("\nSaved CSVs:")
        print(f" - Grid search results: {cv_results_out}")
        print(f" - Fold results: {folds_out}")
        print(f" - Overall metrics: {overall_out}")
        print(f" - Feature importances: {imp_out}")
        print(f" - Confusion matrix (OOF): {cm_out}")
        print(f" - OOF predictions: {oof_out}")
        if roc_path:
            print(f" - ROC plot: {roc_path}")
    except Exception as e:
        print(f"Failed saving CSVs: {e}")


if __name__ == "__main__":
    main()
