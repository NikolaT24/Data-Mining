import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import IsolationForest, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
OUTPUT_DIR = Path("mega_data_mining_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_STATE)


def save_text(name, text):
    with open(OUTPUT_DIR / name, "w", encoding="utf-8") as file:
        file.write(text)


def save_json(name, data):
    with open(OUTPUT_DIR / name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_dataframe(name, dataframe):
    dataframe.to_csv(OUTPUT_DIR / name, index=False)


def make_plot_path(filename):
    return OUTPUT_DIR / filename


def generate_dataset():
    X, y = make_classification(
        n_samples=30000,
        n_features=45,
        n_informative=20,
        n_redundant=12,
        n_classes=2,
        weights=[0.87, 0.13],
        class_sep=1.25,
        flip_y=0.02,
        random_state=RANDOM_STATE
    )

    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    rng = np.random.default_rng(RANDOM_STATE)

    for col in columns[:18]:
        missing_rows = rng.choice(df.index, size=700, replace=False)
        df.loc[missing_rows, col] = np.nan

    for col in columns[18:30]:
        outlier_rows = rng.choice(df.index, size=180, replace=False)
        df.loc[outlier_rows, col] = df.loc[outlier_rows, col] * rng.uniform(4, 9)

    duplicate_rows = df.sample(500, random_state=RANDOM_STATE)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


def explore_dataset(df):
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isnull().sum(),
        "missing_rate": df.isnull().mean(),
        "unique": df.nunique()
    })

    numeric_summary = df.describe().T
    numeric_summary["skew"] = df.skew(numeric_only=True)
    numeric_summary["kurtosis"] = df.kurtosis(numeric_only=True)

    save_dataframe("data_quality_summary.csv", summary.reset_index().rename(columns={"index": "column"}))
    save_dataframe("numeric_summary.csv", numeric_summary.reset_index().rename(columns={"index": "column"}))

    class_distribution = df["target"].value_counts().reset_index()
    class_distribution.columns = ["class", "count"]
    save_dataframe("class_distribution.csv", class_distribution)

    plt.figure(figsize=(6, 4))
    plt.bar(class_distribution["class"].astype(str), class_distribution["count"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(make_plot_path("class_distribution.png"))
    plt.close()

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(make_plot_path("correlation_matrix.png"))
    plt.close()


def prepare_data(df):
    df = df.drop_duplicates()

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X, y, X_train, X_test, y_train, y_test


def build_preprocessor(scaler_name="standard"):
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])


def build_models():
    standard = build_preprocessor("standard")
    robust = build_preprocessor("robust")

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "SGD Classifier": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "KNN": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", KNeighborsClassifier(n_neighbors=9))
        ]),
        "Naive Bayes": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", GaussianNB())
        ]),
        "SVM": Pipeline([
            ("preprocessor", robust),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", ExtraTreesClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE))
        ]),
        "AdaBoost": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", AdaBoostClassifier(random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        "MLP Neural Network": Pipeline([
            ("preprocessor", standard),
            ("selector", SelectKBest(score_func=f_classif, k=30)),
            ("model", MLPClassifier(hidden_layer_sizes=(96, 48, 24), max_iter=500, random_state=RANDOM_STATE))
        ])
    }

    return models


def evaluate_models(models, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    rows = []

    for name, model in models.items():
        start = time.time()

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        elapsed = time.time() - start

        rows.append({
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
            "fit_time": scores["fit_time"].mean(),
            "score_time": scores["score_time"].mean(),
            "total_seconds": elapsed
        })

    results = pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
    save_dataframe("cross_validation_model_comparison.csv", results)

    return results


def evaluate_final_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    report = classification_report(y_test, predictions)
    save_text(f"{name.lower().replace(' ', '_')}_classification_report.txt", report)

    prediction_df = pd.DataFrame({
        "actual": y_test.values,
        "prediction": predictions,
        "probability": probabilities
    })

    save_dataframe(f"{name.lower().replace(' ', '_')}_predictions.csv", prediction_df)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(make_plot_path(f"{name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(make_plot_path(f"{name.lower().replace(' ', '_')}_roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.title(f"{name} Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(make_plot_path(f"{name.lower().replace(' ', '_')}_precision_recall.png"))
    plt.close()

    return metrics, prediction_df


def tune_random_forest(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
    ])

    params = {
        "selector__k": [20, 25, 30, 35, "all"],
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [None, 8, 12, 18],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        pipeline,
        params,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    save_json("random_forest_best_params.json", search.best_params_)

    return search.best_estimator_, search.best_score_


def tune_gradient_boosting(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif)),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    params = {
        "selector__k": [20, 25, 30, 35, "all"],
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__min_samples_split": [2, 5, 10]
    }

    search = RandomizedSearchCV(
        pipeline,
        params,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)
    save_json("gradient_boosting_best_params.json", search.best_params_)

    return search.best_estimator_, search.best_score_


def build_ensemble(best_rf, best_gb, X_train, y_train):
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=30)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    voting = VotingClassifier(
        estimators=[
            ("rf", best_rf),
            ("gb", best_gb),
            ("lr", lr)
        ],
        voting="soft",
        n_jobs=-1
    )

    stacking = StackingClassifier(
        estimators=[
            ("rf", best_rf),
            ("gb", best_gb),
            ("lr", lr)
        ],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
        n_jobs=-1
    )

    voting.fit(X_train, y_train)
    stacking.fit(X_train, y_train)

    return voting, stacking


def threshold_analysis(probabilities, y_test):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        preds = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        })

    df = pd.DataFrame(rows)
    save_dataframe("threshold_analysis.csv", df)

    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.title("Threshold Analysis")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(make_plot_path("threshold_analysis.png"))
    plt.close()

    return df


def run_dimensionality_reduction(X):
    processor = build_preprocessor("standard")
    X_processed = processor.fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_processed)

    explained = pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca_full.explained_variance_ratio_)
    })

    save_dataframe("pca_explained_variance.csv", explained)

    plt.figure(figsize=(8, 5))
    plt.plot(explained["component"], explained["cumulative_variance"])
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Component")
    plt.ylabel("Cumulative Variance")
    plt.tight_layout()
    plt.savefig(make_plot_path("pca_cumulative_variance.png"))
    plt.close()

    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca_2.fit_transform(X_processed)

    reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
    save_dataframe("pca_2d_projection.csv", reduced_df)

    return explained, reduced_df


def run_clustering(X):
    processor = build_preprocessor("standard")
    X_processed = processor.fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed)

    rows = []

    for k in range(2, 13):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_processed)

        rows.append({
            "algorithm": "KMeans",
            "clusters": k,
            "silhouette": silhouette_score(X_processed, labels),
            "davies_bouldin": davies_bouldin_score(X_processed, labels),
            "calinski_harabasz": calinski_harabasz_score(X_processed, labels)
        })

    for k in range(2, 13):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_processed)

        rows.append({
            "algorithm": "Agglomerative",
            "clusters": k,
            "silhouette": silhouette_score(X_processed, labels),
            "davies_bouldin": davies_bouldin_score(X_processed, labels),
            "calinski_harabasz": calinski_harabasz_score(X_processed, labels)
        })

    for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
        model = DBSCAN(eps=eps, min_samples=10)
        labels = model.fit_predict(X_processed)
        unique_labels = set(labels)

        if len(unique_labels) > 1 and len(unique_labels) < len(X):
            rows.append({
                "algorithm": "DBSCAN",
                "clusters": len(unique_labels),
                "silhouette": silhouette_score(X_processed, labels),
                "davies_bouldin": davies_bouldin_score(X_processed, labels),
                "calinski_harabasz": calinski_harabasz_score(X_processed, labels)
            })

    clustering_df = pd.DataFrame(rows)
    save_dataframe("clustering_comparison.csv", clustering_df)

    best = clustering_df.sort_values(by="silhouette", ascending=False).iloc[0]

    if best["algorithm"] == "KMeans":
        best_cluster_model = KMeans(n_clusters=int(best["clusters"]), random_state=RANDOM_STATE, n_init=10)
    elif best["algorithm"] == "Agglomerative":
        best_cluster_model = AgglomerativeClustering(n_clusters=int(best["clusters"]))
    else:
        best_cluster_model = DBSCAN(eps=2.5, min_samples=10)

    labels = best_cluster_model.fit_predict(X_processed)

    cluster_df = pd.DataFrame(X.copy())
    cluster_df["cluster"] = labels
    cluster_summary = cluster_df.groupby("cluster").mean(numeric_only=True)

    save_dataframe("cluster_assignments.csv", cluster_df)
    save_dataframe("cluster_summary.csv", cluster_summary.reset_index())

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, alpha=0.5)
    plt.title("Best Clustering Result")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(make_plot_path("best_clustering.png"))
    plt.close()

    return clustering_df, cluster_df


def run_outlier_detection(X):
    processor = build_preprocessor("standard")
    X_processed = processor.fit_transform(X)

    isolation = IsolationForest(contamination=0.08, random_state=RANDOM_STATE)
    iso_labels = isolation.fit_predict(X_processed)
    iso_scores = isolation.decision_function(X_processed)

    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.08)
    lof_labels = lof.fit_predict(X_processed)

    outlier_df = X.copy()
    outlier_df["isolation_label"] = iso_labels
    outlier_df["isolation_score"] = iso_scores
    outlier_df["lof_label"] = lof_labels
    outlier_df["possible_outlier"] = ((iso_labels == -1) | (lof_labels == -1)).astype(int)

    save_dataframe("outlier_detection_results.csv", outlier_df)

    return outlier_df


def feature_importance_report(model, X):
    try:
        selector = model.named_steps.get("selector")
        estimator = model.named_steps.get("model")

        if selector is not None:
            mask = selector.get_support()
            selected_columns = X.columns[mask]
        else:
            selected_columns = X.columns

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).ravel()
        else:
            return None

        importance_df = pd.DataFrame({
            "feature": selected_columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        save_dataframe("feature_importance.csv", importance_df)

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["feature"], importance_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(make_plot_path("feature_importance.png"))
        plt.close()

        return importance_df
    except Exception:
        return None



def segment_report_1(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 1,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_1.csv", report)
    return report


def transformation_block_1(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_1"] = transformed[column].abs()
            transformed[f"{column}_squared_1"] = transformed[column] ** 2
            transformed[f"{column}_rank_1"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_1(df):
    audit = {
        "block": 1,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_1.json", audit)
    return audit


def segment_report_2(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 2,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_2.csv", report)
    return report


def transformation_block_2(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_2"] = transformed[column].abs()
            transformed[f"{column}_squared_2"] = transformed[column] ** 2
            transformed[f"{column}_rank_2"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_2(df):
    audit = {
        "block": 2,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_2.json", audit)
    return audit


def segment_report_3(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 3,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_3.csv", report)
    return report


def transformation_block_3(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_3"] = transformed[column].abs()
            transformed[f"{column}_squared_3"] = transformed[column] ** 2
            transformed[f"{column}_rank_3"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_3(df):
    audit = {
        "block": 3,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_3.json", audit)
    return audit


def segment_report_4(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 4,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_4.csv", report)
    return report


def transformation_block_4(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_4"] = transformed[column].abs()
            transformed[f"{column}_squared_4"] = transformed[column] ** 2
            transformed[f"{column}_rank_4"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_4(df):
    audit = {
        "block": 4,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_4.json", audit)
    return audit


def segment_report_5(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 5,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_5.csv", report)
    return report


def transformation_block_5(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_5"] = transformed[column].abs()
            transformed[f"{column}_squared_5"] = transformed[column] ** 2
            transformed[f"{column}_rank_5"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_5(df):
    audit = {
        "block": 5,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_5.json", audit)
    return audit


def segment_report_6(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 6,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_6.csv", report)
    return report


def transformation_block_6(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_6"] = transformed[column].abs()
            transformed[f"{column}_squared_6"] = transformed[column] ** 2
            transformed[f"{column}_rank_6"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_6(df):
    audit = {
        "block": 6,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_6.json", audit)
    return audit


def segment_report_7(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 7,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_7.csv", report)
    return report


def transformation_block_7(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_7"] = transformed[column].abs()
            transformed[f"{column}_squared_7"] = transformed[column] ** 2
            transformed[f"{column}_rank_7"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_7(df):
    audit = {
        "block": 7,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_7.json", audit)
    return audit


def segment_report_8(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 8,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_8.csv", report)
    return report


def transformation_block_8(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_8"] = transformed[column].abs()
            transformed[f"{column}_squared_8"] = transformed[column] ** 2
            transformed[f"{column}_rank_8"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_8(df):
    audit = {
        "block": 8,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_8.json", audit)
    return audit


def segment_report_9(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 9,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_9.csv", report)
    return report


def transformation_block_9(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_9"] = transformed[column].abs()
            transformed[f"{column}_squared_9"] = transformed[column] ** 2
            transformed[f"{column}_rank_9"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_9(df):
    audit = {
        "block": 9,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_9.json", audit)
    return audit


def segment_report_10(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 10,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_10.csv", report)
    return report


def transformation_block_10(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_10"] = transformed[column].abs()
            transformed[f"{column}_squared_10"] = transformed[column] ** 2
            transformed[f"{column}_rank_10"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_10(df):
    audit = {
        "block": 10,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_10.json", audit)
    return audit


def segment_report_11(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 11,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_11.csv", report)
    return report


def transformation_block_11(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_11"] = transformed[column].abs()
            transformed[f"{column}_squared_11"] = transformed[column] ** 2
            transformed[f"{column}_rank_11"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_11(df):
    audit = {
        "block": 11,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_11.json", audit)
    return audit


def segment_report_12(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 12,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_12.csv", report)
    return report


def transformation_block_12(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_12"] = transformed[column].abs()
            transformed[f"{column}_squared_12"] = transformed[column] ** 2
            transformed[f"{column}_rank_12"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_12(df):
    audit = {
        "block": 12,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_12.json", audit)
    return audit


def segment_report_13(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 13,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_13.csv", report)
    return report


def transformation_block_13(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_13"] = transformed[column].abs()
            transformed[f"{column}_squared_13"] = transformed[column] ** 2
            transformed[f"{column}_rank_13"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_13(df):
    audit = {
        "block": 13,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_13.json", audit)
    return audit


def segment_report_14(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 14,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_14.csv", report)
    return report


def transformation_block_14(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_14"] = transformed[column].abs()
            transformed[f"{column}_squared_14"] = transformed[column] ** 2
            transformed[f"{column}_rank_14"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_14(df):
    audit = {
        "block": 14,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_14.json", audit)
    return audit


def segment_report_15(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 15,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_15.csv", report)
    return report


def transformation_block_15(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_15"] = transformed[column].abs()
            transformed[f"{column}_squared_15"] = transformed[column] ** 2
            transformed[f"{column}_rank_15"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_15(df):
    audit = {
        "block": 15,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_15.json", audit)
    return audit


def segment_report_16(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 16,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_16.csv", report)
    return report


def transformation_block_16(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_16"] = transformed[column].abs()
            transformed[f"{column}_squared_16"] = transformed[column] ** 2
            transformed[f"{column}_rank_16"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_16(df):
    audit = {
        "block": 16,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_16.json", audit)
    return audit


def segment_report_17(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 17,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_17.csv", report)
    return report


def transformation_block_17(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_17"] = transformed[column].abs()
            transformed[f"{column}_squared_17"] = transformed[column] ** 2
            transformed[f"{column}_rank_17"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_17(df):
    audit = {
        "block": 17,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_17.json", audit)
    return audit


def segment_report_18(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 18,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_18.csv", report)
    return report


def transformation_block_18(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_18"] = transformed[column].abs()
            transformed[f"{column}_squared_18"] = transformed[column] ** 2
            transformed[f"{column}_rank_18"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_18(df):
    audit = {
        "block": 18,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_18.json", audit)
    return audit


def segment_report_19(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 19,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_19.csv", report)
    return report


def transformation_block_19(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_19"] = transformed[column].abs()
            transformed[f"{column}_squared_19"] = transformed[column] ** 2
            transformed[f"{column}_rank_19"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_19(df):
    audit = {
        "block": 19,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_19.json", audit)
    return audit


def segment_report_20(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 20,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_20.csv", report)
    return report


def transformation_block_20(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_20"] = transformed[column].abs()
            transformed[f"{column}_squared_20"] = transformed[column] ** 2
            transformed[f"{column}_rank_20"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_20(df):
    audit = {
        "block": 20,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_20.json", audit)
    return audit


def segment_report_21(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 21,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_21.csv", report)
    return report


def transformation_block_21(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_21"] = transformed[column].abs()
            transformed[f"{column}_squared_21"] = transformed[column] ** 2
            transformed[f"{column}_rank_21"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_21(df):
    audit = {
        "block": 21,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_21.json", audit)
    return audit


def segment_report_22(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 22,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_22.csv", report)
    return report


def transformation_block_22(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_22"] = transformed[column].abs()
            transformed[f"{column}_squared_22"] = transformed[column] ** 2
            transformed[f"{column}_rank_22"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_22(df):
    audit = {
        "block": 22,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_22.json", audit)
    return audit


def segment_report_23(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 23,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_23.csv", report)
    return report


def transformation_block_23(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_23"] = transformed[column].abs()
            transformed[f"{column}_squared_23"] = transformed[column] ** 2
            transformed[f"{column}_rank_23"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_23(df):
    audit = {
        "block": 23,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_23.json", audit)
    return audit


def segment_report_24(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 24,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_24.csv", report)
    return report


def transformation_block_24(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_24"] = transformed[column].abs()
            transformed[f"{column}_squared_24"] = transformed[column] ** 2
            transformed[f"{column}_rank_24"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_24(df):
    audit = {
        "block": 24,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_24.json", audit)
    return audit


def segment_report_25(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 25,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_25.csv", report)
    return report


def transformation_block_25(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_25"] = transformed[column].abs()
            transformed[f"{column}_squared_25"] = transformed[column] ** 2
            transformed[f"{column}_rank_25"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_25(df):
    audit = {
        "block": 25,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_25.json", audit)
    return audit


def segment_report_26(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 26,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_26.csv", report)
    return report


def transformation_block_26(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_26"] = transformed[column].abs()
            transformed[f"{column}_squared_26"] = transformed[column] ** 2
            transformed[f"{column}_rank_26"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_26(df):
    audit = {
        "block": 26,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_26.json", audit)
    return audit


def segment_report_27(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 27,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_27.csv", report)
    return report


def transformation_block_27(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_27"] = transformed[column].abs()
            transformed[f"{column}_squared_27"] = transformed[column] ** 2
            transformed[f"{column}_rank_27"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_27(df):
    audit = {
        "block": 27,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_27.json", audit)
    return audit


def segment_report_28(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 28,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_28.csv", report)
    return report


def transformation_block_28(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_28"] = transformed[column].abs()
            transformed[f"{column}_squared_28"] = transformed[column] ** 2
            transformed[f"{column}_rank_28"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_28(df):
    audit = {
        "block": 28,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_28.json", audit)
    return audit


def segment_report_29(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 29,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_29.csv", report)
    return report


def transformation_block_29(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_29"] = transformed[column].abs()
            transformed[f"{column}_squared_29"] = transformed[column] ** 2
            transformed[f"{column}_rank_29"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_29(df):
    audit = {
        "block": 29,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_29.json", audit)
    return audit


def segment_report_30(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 30,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_30.csv", report)
    return report


def transformation_block_30(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_30"] = transformed[column].abs()
            transformed[f"{column}_squared_30"] = transformed[column] ** 2
            transformed[f"{column}_rank_30"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_30(df):
    audit = {
        "block": 30,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_30.json", audit)
    return audit


def segment_report_31(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 31,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_31.csv", report)
    return report


def transformation_block_31(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_31"] = transformed[column].abs()
            transformed[f"{column}_squared_31"] = transformed[column] ** 2
            transformed[f"{column}_rank_31"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_31(df):
    audit = {
        "block": 31,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_31.json", audit)
    return audit


def segment_report_32(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 32,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_32.csv", report)
    return report


def transformation_block_32(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_32"] = transformed[column].abs()
            transformed[f"{column}_squared_32"] = transformed[column] ** 2
            transformed[f"{column}_rank_32"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_32(df):
    audit = {
        "block": 32,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_32.json", audit)
    return audit


def segment_report_33(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 33,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_33.csv", report)
    return report


def transformation_block_33(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_33"] = transformed[column].abs()
            transformed[f"{column}_squared_33"] = transformed[column] ** 2
            transformed[f"{column}_rank_33"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_33(df):
    audit = {
        "block": 33,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_33.json", audit)
    return audit


def segment_report_34(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 34,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_34.csv", report)
    return report


def transformation_block_34(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_34"] = transformed[column].abs()
            transformed[f"{column}_squared_34"] = transformed[column] ** 2
            transformed[f"{column}_rank_34"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_34(df):
    audit = {
        "block": 34,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_34.json", audit)
    return audit


def segment_report_35(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 35,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_35.csv", report)
    return report


def transformation_block_35(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_35"] = transformed[column].abs()
            transformed[f"{column}_squared_35"] = transformed[column] ** 2
            transformed[f"{column}_rank_35"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_35(df):
    audit = {
        "block": 35,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_35.json", audit)
    return audit


def segment_report_36(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 36,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_36.csv", report)
    return report


def transformation_block_36(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_36"] = transformed[column].abs()
            transformed[f"{column}_squared_36"] = transformed[column] ** 2
            transformed[f"{column}_rank_36"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_36(df):
    audit = {
        "block": 36,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_36.json", audit)
    return audit


def segment_report_37(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 37,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_37.csv", report)
    return report


def transformation_block_37(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_37"] = transformed[column].abs()
            transformed[f"{column}_squared_37"] = transformed[column] ** 2
            transformed[f"{column}_rank_37"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_37(df):
    audit = {
        "block": 37,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_37.json", audit)
    return audit


def segment_report_38(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 38,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_38.csv", report)
    return report


def transformation_block_38(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_38"] = transformed[column].abs()
            transformed[f"{column}_squared_38"] = transformed[column] ** 2
            transformed[f"{column}_rank_38"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_38(df):
    audit = {
        "block": 38,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_38.json", audit)
    return audit


def segment_report_39(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 39,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_39.csv", report)
    return report


def transformation_block_39(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_39"] = transformed[column].abs()
            transformed[f"{column}_squared_39"] = transformed[column] ** 2
            transformed[f"{column}_rank_39"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_39(df):
    audit = {
        "block": 39,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_39.json", audit)
    return audit


def segment_report_40(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 40,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_40.csv", report)
    return report


def transformation_block_40(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_40"] = transformed[column].abs()
            transformed[f"{column}_squared_40"] = transformed[column] ** 2
            transformed[f"{column}_rank_40"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_40(df):
    audit = {
        "block": 40,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_40.json", audit)
    return audit


def segment_report_41(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 41,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_41.csv", report)
    return report


def transformation_block_41(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_41"] = transformed[column].abs()
            transformed[f"{column}_squared_41"] = transformed[column] ** 2
            transformed[f"{column}_rank_41"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_41(df):
    audit = {
        "block": 41,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_41.json", audit)
    return audit


def segment_report_42(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 42,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_42.csv", report)
    return report


def transformation_block_42(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_42"] = transformed[column].abs()
            transformed[f"{column}_squared_42"] = transformed[column] ** 2
            transformed[f"{column}_rank_42"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_42(df):
    audit = {
        "block": 42,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_42.json", audit)
    return audit


def segment_report_43(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 43,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_43.csv", report)
    return report


def transformation_block_43(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_43"] = transformed[column].abs()
            transformed[f"{column}_squared_43"] = transformed[column] ** 2
            transformed[f"{column}_rank_43"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_43(df):
    audit = {
        "block": 43,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_43.json", audit)
    return audit


def segment_report_44(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 44,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_44.csv", report)
    return report


def transformation_block_44(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_44"] = transformed[column].abs()
            transformed[f"{column}_squared_44"] = transformed[column] ** 2
            transformed[f"{column}_rank_44"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_44(df):
    audit = {
        "block": 44,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_44.json", audit)
    return audit


def segment_report_45(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 45,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_45.csv", report)
    return report


def transformation_block_45(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_45"] = transformed[column].abs()
            transformed[f"{column}_squared_45"] = transformed[column] ** 2
            transformed[f"{column}_rank_45"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_45(df):
    audit = {
        "block": 45,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_45.json", audit)
    return audit


def segment_report_46(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 46,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_46.csv", report)
    return report


def transformation_block_46(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_46"] = transformed[column].abs()
            transformed[f"{column}_squared_46"] = transformed[column] ** 2
            transformed[f"{column}_rank_46"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_46(df):
    audit = {
        "block": 46,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_46.json", audit)
    return audit


def segment_report_47(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 47,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_47.csv", report)
    return report


def transformation_block_47(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_47"] = transformed[column].abs()
            transformed[f"{column}_squared_47"] = transformed[column] ** 2
            transformed[f"{column}_rank_47"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_47(df):
    audit = {
        "block": 47,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_47.json", audit)
    return audit


def segment_report_48(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 48,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_48.csv", report)
    return report


def transformation_block_48(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_48"] = transformed[column].abs()
            transformed[f"{column}_squared_48"] = transformed[column] ** 2
            transformed[f"{column}_rank_48"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_48(df):
    audit = {
        "block": 48,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_48.json", audit)
    return audit


def segment_report_49(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 49,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_49.csv", report)
    return report


def transformation_block_49(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_49"] = transformed[column].abs()
            transformed[f"{column}_squared_49"] = transformed[column] ** 2
            transformed[f"{column}_rank_49"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_49(df):
    audit = {
        "block": 49,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_49.json", audit)
    return audit


def segment_report_50(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 50,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_50.csv", report)
    return report


def transformation_block_50(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_50"] = transformed[column].abs()
            transformed[f"{column}_squared_50"] = transformed[column] ** 2
            transformed[f"{column}_rank_50"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_50(df):
    audit = {
        "block": 50,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_50.json", audit)
    return audit


def segment_report_51(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 51,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_51.csv", report)
    return report


def transformation_block_51(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_51"] = transformed[column].abs()
            transformed[f"{column}_squared_51"] = transformed[column] ** 2
            transformed[f"{column}_rank_51"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_51(df):
    audit = {
        "block": 51,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_51.json", audit)
    return audit


def segment_report_52(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 52,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_52.csv", report)
    return report


def transformation_block_52(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_52"] = transformed[column].abs()
            transformed[f"{column}_squared_52"] = transformed[column] ** 2
            transformed[f"{column}_rank_52"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_52(df):
    audit = {
        "block": 52,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_52.json", audit)
    return audit


def segment_report_53(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 53,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_53.csv", report)
    return report


def transformation_block_53(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_53"] = transformed[column].abs()
            transformed[f"{column}_squared_53"] = transformed[column] ** 2
            transformed[f"{column}_rank_53"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_53(df):
    audit = {
        "block": 53,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_53.json", audit)
    return audit


def segment_report_54(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 54,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_54.csv", report)
    return report


def transformation_block_54(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_54"] = transformed[column].abs()
            transformed[f"{column}_squared_54"] = transformed[column] ** 2
            transformed[f"{column}_rank_54"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_54(df):
    audit = {
        "block": 54,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_54.json", audit)
    return audit


def segment_report_55(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 55,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_55.csv", report)
    return report


def transformation_block_55(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_55"] = transformed[column].abs()
            transformed[f"{column}_squared_55"] = transformed[column] ** 2
            transformed[f"{column}_rank_55"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_55(df):
    audit = {
        "block": 55,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_55.json", audit)
    return audit


def segment_report_56(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 56,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_56.csv", report)
    return report


def transformation_block_56(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_56"] = transformed[column].abs()
            transformed[f"{column}_squared_56"] = transformed[column] ** 2
            transformed[f"{column}_rank_56"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_56(df):
    audit = {
        "block": 56,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_56.json", audit)
    return audit


def segment_report_57(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 57,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_57.csv", report)
    return report


def transformation_block_57(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_57"] = transformed[column].abs()
            transformed[f"{column}_squared_57"] = transformed[column] ** 2
            transformed[f"{column}_rank_57"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_57(df):
    audit = {
        "block": 57,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_57.json", audit)
    return audit


def segment_report_58(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 58,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_58.csv", report)
    return report


def transformation_block_58(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_58"] = transformed[column].abs()
            transformed[f"{column}_squared_58"] = transformed[column] ** 2
            transformed[f"{column}_rank_58"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_58(df):
    audit = {
        "block": 58,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_58.json", audit)
    return audit


def segment_report_59(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 59,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_59.csv", report)
    return report


def transformation_block_59(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_59"] = transformed[column].abs()
            transformed[f"{column}_squared_59"] = transformed[column] ** 2
            transformed[f"{column}_rank_59"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_59(df):
    audit = {
        "block": 59,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_59.json", audit)
    return audit


def segment_report_60(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 60,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_60.csv", report)
    return report


def transformation_block_60(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_60"] = transformed[column].abs()
            transformed[f"{column}_squared_60"] = transformed[column] ** 2
            transformed[f"{column}_rank_60"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_60(df):
    audit = {
        "block": 60,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_60.json", audit)
    return audit


def segment_report_61(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 61,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_61.csv", report)
    return report


def transformation_block_61(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_61"] = transformed[column].abs()
            transformed[f"{column}_squared_61"] = transformed[column] ** 2
            transformed[f"{column}_rank_61"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_61(df):
    audit = {
        "block": 61,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_61.json", audit)
    return audit


def segment_report_62(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 62,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_62.csv", report)
    return report


def transformation_block_62(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_62"] = transformed[column].abs()
            transformed[f"{column}_squared_62"] = transformed[column] ** 2
            transformed[f"{column}_rank_62"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_62(df):
    audit = {
        "block": 62,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_62.json", audit)
    return audit


def segment_report_63(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 63,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_63.csv", report)
    return report


def transformation_block_63(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_63"] = transformed[column].abs()
            transformed[f"{column}_squared_63"] = transformed[column] ** 2
            transformed[f"{column}_rank_63"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_63(df):
    audit = {
        "block": 63,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_63.json", audit)
    return audit


def segment_report_64(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 64,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_64.csv", report)
    return report


def transformation_block_64(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_64"] = transformed[column].abs()
            transformed[f"{column}_squared_64"] = transformed[column] ** 2
            transformed[f"{column}_rank_64"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_64(df):
    audit = {
        "block": 64,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_64.json", audit)
    return audit


def segment_report_65(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 65,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_65.csv", report)
    return report


def transformation_block_65(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_65"] = transformed[column].abs()
            transformed[f"{column}_squared_65"] = transformed[column] ** 2
            transformed[f"{column}_rank_65"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_65(df):
    audit = {
        "block": 65,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_65.json", audit)
    return audit


def segment_report_66(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 66,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_66.csv", report)
    return report


def transformation_block_66(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_66"] = transformed[column].abs()
            transformed[f"{column}_squared_66"] = transformed[column] ** 2
            transformed[f"{column}_rank_66"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_66(df):
    audit = {
        "block": 66,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_66.json", audit)
    return audit


def segment_report_67(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 67,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_67.csv", report)
    return report


def transformation_block_67(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_67"] = transformed[column].abs()
            transformed[f"{column}_squared_67"] = transformed[column] ** 2
            transformed[f"{column}_rank_67"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_67(df):
    audit = {
        "block": 67,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_67.json", audit)
    return audit


def segment_report_68(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 68,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_68.csv", report)
    return report


def transformation_block_68(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_68"] = transformed[column].abs()
            transformed[f"{column}_squared_68"] = transformed[column] ** 2
            transformed[f"{column}_rank_68"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_68(df):
    audit = {
        "block": 68,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_68.json", audit)
    return audit


def segment_report_69(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 69,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_69.csv", report)
    return report


def transformation_block_69(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_69"] = transformed[column].abs()
            transformed[f"{column}_squared_69"] = transformed[column] ** 2
            transformed[f"{column}_rank_69"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_69(df):
    audit = {
        "block": 69,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_69.json", audit)
    return audit


def segment_report_70(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 70,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_70.csv", report)
    return report


def transformation_block_70(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_70"] = transformed[column].abs()
            transformed[f"{column}_squared_70"] = transformed[column] ** 2
            transformed[f"{column}_rank_70"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_70(df):
    audit = {
        "block": 70,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_70.json", audit)
    return audit


def segment_report_71(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 71,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_71.csv", report)
    return report


def transformation_block_71(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_71"] = transformed[column].abs()
            transformed[f"{column}_squared_71"] = transformed[column] ** 2
            transformed[f"{column}_rank_71"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_71(df):
    audit = {
        "block": 71,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_71.json", audit)
    return audit


def segment_report_72(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 72,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_72.csv", report)
    return report


def transformation_block_72(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_72"] = transformed[column].abs()
            transformed[f"{column}_squared_72"] = transformed[column] ** 2
            transformed[f"{column}_rank_72"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_72(df):
    audit = {
        "block": 72,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_72.json", audit)
    return audit


def segment_report_73(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 73,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_73.csv", report)
    return report


def transformation_block_73(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_73"] = transformed[column].abs()
            transformed[f"{column}_squared_73"] = transformed[column] ** 2
            transformed[f"{column}_rank_73"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_73(df):
    audit = {
        "block": 73,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_73.json", audit)
    return audit


def segment_report_74(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 74,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_74.csv", report)
    return report


def transformation_block_74(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_74"] = transformed[column].abs()
            transformed[f"{column}_squared_74"] = transformed[column] ** 2
            transformed[f"{column}_rank_74"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_74(df):
    audit = {
        "block": 74,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_74.json", audit)
    return audit


def segment_report_75(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 75,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_75.csv", report)
    return report


def transformation_block_75(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_75"] = transformed[column].abs()
            transformed[f"{column}_squared_75"] = transformed[column] ** 2
            transformed[f"{column}_rank_75"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_75(df):
    audit = {
        "block": 75,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_75.json", audit)
    return audit


def segment_report_76(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 76,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_76.csv", report)
    return report


def transformation_block_76(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_76"] = transformed[column].abs()
            transformed[f"{column}_squared_76"] = transformed[column] ** 2
            transformed[f"{column}_rank_76"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_76(df):
    audit = {
        "block": 76,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_76.json", audit)
    return audit


def segment_report_77(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 77,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_77.csv", report)
    return report


def transformation_block_77(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_77"] = transformed[column].abs()
            transformed[f"{column}_squared_77"] = transformed[column] ** 2
            transformed[f"{column}_rank_77"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_77(df):
    audit = {
        "block": 77,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_77.json", audit)
    return audit


def segment_report_78(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 78,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_78.csv", report)
    return report


def transformation_block_78(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_78"] = transformed[column].abs()
            transformed[f"{column}_squared_78"] = transformed[column] ** 2
            transformed[f"{column}_rank_78"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_78(df):
    audit = {
        "block": 78,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_78.json", audit)
    return audit


def segment_report_79(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 79,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_79.csv", report)
    return report


def transformation_block_79(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_79"] = transformed[column].abs()
            transformed[f"{column}_squared_79"] = transformed[column] ** 2
            transformed[f"{column}_rank_79"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_79(df):
    audit = {
        "block": 79,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_79.json", audit)
    return audit


def segment_report_80(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 80,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_80.csv", report)
    return report


def transformation_block_80(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_80"] = transformed[column].abs()
            transformed[f"{column}_squared_80"] = transformed[column] ** 2
            transformed[f"{column}_rank_80"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_80(df):
    audit = {
        "block": 80,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_80.json", audit)
    return audit


def segment_report_81(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 81,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_81.csv", report)
    return report


def transformation_block_81(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_81"] = transformed[column].abs()
            transformed[f"{column}_squared_81"] = transformed[column] ** 2
            transformed[f"{column}_rank_81"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_81(df):
    audit = {
        "block": 81,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_81.json", audit)
    return audit


def segment_report_82(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 82,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_82.csv", report)
    return report


def transformation_block_82(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_82"] = transformed[column].abs()
            transformed[f"{column}_squared_82"] = transformed[column] ** 2
            transformed[f"{column}_rank_82"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_82(df):
    audit = {
        "block": 82,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_82.json", audit)
    return audit


def segment_report_83(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 83,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_83.csv", report)
    return report


def transformation_block_83(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_83"] = transformed[column].abs()
            transformed[f"{column}_squared_83"] = transformed[column] ** 2
            transformed[f"{column}_rank_83"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_83(df):
    audit = {
        "block": 83,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_83.json", audit)
    return audit


def segment_report_84(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 84,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_84.csv", report)
    return report


def transformation_block_84(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_84"] = transformed[column].abs()
            transformed[f"{column}_squared_84"] = transformed[column] ** 2
            transformed[f"{column}_rank_84"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_84(df):
    audit = {
        "block": 84,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_84.json", audit)
    return audit


def segment_report_85(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    report_rows = []
    for column in numeric_columns:
        series = df[column]
        report_rows.append({
            "segment": 85,
            "column": column,
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "missing": series.isnull().sum()
        })
    report = pd.DataFrame(report_rows)
    save_dataframe("segment_report_85.csv", report)
    return report


def transformation_block_85(df):
    transformed = df.copy()
    numeric_columns = transformed.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if column != "target":
            transformed[f"{column}_abs_85"] = transformed[column].abs()
            transformed[f"{column}_squared_85"] = transformed[column] ** 2
            transformed[f"{column}_rank_85"] = transformed[column].rank(method="average")
            break
    return transformed


def audit_block_85(df):
    audit = {
        "block": 85,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_total": int(df.duplicated().sum())
    }
    save_json("audit_block_85.json", audit)
    return audit

def create_final_report(metadata):
    lines = []
    lines.append("MEGA DATA MINING 1000 LINE REPORT")
    lines.append("=" * 80)
    lines.append(f"Created: {datetime.now()}")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Files generated in mega_data_mining_outputs/")
    lines.append("=" * 80)

    for file in sorted(OUTPUT_DIR.iterdir()):
        lines.append(file.name)

    save_text("README_REPORT.txt", "\n".join(lines))


def main():
    df = generate_dataset()
    save_dataframe("raw_dataset.csv", df)

    explore_dataset(df)

    for block_id in range(1, 6):
        globals()[f"audit_block_{block_id}"](df)
        globals()[f"segment_report_{block_id}"](df)

    df_enriched = df.copy()
    for block_id in range(1, 4):
        df_enriched = globals()[f"transformation_block_{block_id}"](df_enriched)

    save_dataframe("enriched_dataset_sample.csv", df_enriched.head(1000))

    X, y, X_train, X_test, y_train, y_test = prepare_data(df)

    save_dataframe("clean_features.csv", X)
    save_dataframe("clean_target.csv", pd.DataFrame({"target": y}))

    run_dimensionality_reduction(X)

    models = build_models()
    cv_results = evaluate_models(models, X_train, y_train)

    best_model_name = cv_results.iloc[0]["model"]
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    base_metrics, base_predictions = evaluate_final_model(
        best_model_name,
        best_model,
        X_test,
        y_test
    )

    best_rf, rf_cv_score = tune_random_forest(X_train, y_train)

    rf_metrics, rf_predictions = evaluate_final_model(
        "Tuned Random Forest",
        best_rf,
        X_test,
        y_test
    )

    best_gb, gb_cv_score = tune_gradient_boosting(X_train, y_train)

    gb_metrics, gb_predictions = evaluate_final_model(
        "Tuned Gradient Boosting",
        best_gb,
        X_test,
        y_test
    )

    voting, stacking = build_ensemble(best_rf, best_gb, X_train, y_train)

    voting_metrics, voting_predictions = evaluate_final_model(
        "Voting Ensemble",
        voting,
        X_test,
        y_test
    )

    stacking_metrics, stacking_predictions = evaluate_final_model(
        "Stacking Ensemble",
        stacking,
        X_test,
        y_test
    )

    final_results = pd.DataFrame([
        base_metrics,
        rf_metrics,
        gb_metrics,
        voting_metrics,
        stacking_metrics
    ]).sort_values(by="roc_auc", ascending=False)

    save_dataframe("final_model_results.csv", final_results)

    best_final_name = final_results.iloc[0]["model"]

    model_map = {
        best_model_name: best_model,
        "Tuned Random Forest": best_rf,
        "Tuned Gradient Boosting": best_gb,
        "Voting Ensemble": voting,
        "Stacking Ensemble": stacking
    }

    final_model = model_map[best_final_name]

    final_probs = final_model.predict_proba(X_test)[:, 1]
    threshold_analysis(final_probs, y_test)

    run_clustering(X)
    run_outlier_detection(X)
    feature_importance_report(final_model, X)

    joblib.dump(final_model, OUTPUT_DIR / "best_final_model.pkl")
    joblib.dump(best_rf, OUTPUT_DIR / "best_random_forest.pkl")
    joblib.dump(best_gb, OUTPUT_DIR / "best_gradient_boosting.pkl")
    joblib.dump(voting, OUTPUT_DIR / "voting_ensemble.pkl")
    joblib.dump(stacking, OUTPUT_DIR / "stacking_ensemble.pkl")

    metadata = {
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "best_cv_model": best_model_name,
        "best_final_model": best_final_name,
        "random_forest_cv_auc": float(rf_cv_score),
        "gradient_boosting_cv_auc": float(gb_cv_score),
        "best_test_roc_auc": float(final_results.iloc[0]["roc_auc"])
    }

    save_json("project_metadata.json", metadata)
    create_final_report(metadata)

    print("Done.")
    print("Best final model:", best_final_name)
    print(final_results)
    print("Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
