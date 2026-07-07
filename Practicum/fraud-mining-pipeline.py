import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import silhouette_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest, f_classif

X, y = make_classification(
    n_samples=15000,
    n_features=25,
    n_informative=12,
    n_redundant=6,
    n_repeated=0,
    n_classes=2,
    weights=[0.92, 0.08],
    class_sep=1.3,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
dataset = pd.DataFrame(X, columns=feature_names)
dataset["target"] = y

rng = np.random.default_rng(42)

for col in feature_names[:8]:
    missing_index = rng.choice(dataset.index, size=300, replace=False)
    dataset.loc[missing_index, col] = np.nan

duplicate_rows = dataset.sample(150, random_state=42)
dataset = pd.concat([dataset, duplicate_rows], ignore_index=True)

print("Initial Shape")
print(dataset.shape)

print("\nClass Distribution")
print(dataset["target"].value_counts())

print("\nMissing Values")
print(dataset.isnull().sum())

print("\nDuplicate Rows")
print(dataset.duplicated().sum())

dataset = dataset.drop_duplicates()

X = dataset.drop("target", axis=1)
y = dataset["target"]

numeric_columns = X.columns.tolist()

summary = X.describe().T
summary["missing"] = X.isnull().sum()
summary["skew"] = X.skew()
summary["kurtosis"] = X.kurtosis()

print("\nSummary")
print(summary)

correlation_matrix = dataset.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, aspect="auto")
plt.colorbar()
plt.title("Correlation Matrix")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

preprocessor = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, alpha=0.5)
plt.title("PCA View of Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

selector = SelectKBest(score_func=f_classif, k=15)
selector.fit(X_train_processed, y_train)

selected_features = X.columns[selector.get_support()]
feature_scores = pd.DataFrame(
    {
        "Feature": X.columns,
        "Score": selector.scores_,
    }
).sort_values(by="Score", ascending=False)

print("\nSelected Features")
print(selected_features.tolist())

print("\nFeature Scores")
print(feature_scores)

plt.figure(figsize=(10, 6))
plt.barh(feature_scores["Feature"], feature_scores["Score"])
plt.gca().invert_yaxis()
plt.title("Feature Selection Scores")
plt.xlabel("ANOVA F-score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

models = {
    "Logistic Regression": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "Support Vector Machine": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    ),
    "Decision Tree": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=15)),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    ),
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

comparison_rows = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )

    comparison_rows.append(
        {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean(),
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values(by="ROC_AUC", ascending=False)

print("\nModel Comparison")
print(comparison_df)

best_base_name = comparison_df.iloc[0]["Model"]
best_base_model = models[best_base_name]
best_base_model.fit(X_train, y_train)

base_predictions = best_base_model.predict(X_test)
base_probabilities = best_base_model.predict_proba(X_test)[:, 1]

print("\nBest Base Model")
print(best_base_name)

print("\nBase Model Confusion Matrix")
print(confusion_matrix(y_test, base_predictions))

print("\nBase Model Report")
print(classification_report(y_test, base_predictions))

print("\nBase Model ROC AUC")
print(roc_auc_score(y_test, base_probabilities))

rf_grid = {
    "selector__k": [10, 15, 20, "all"],
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [None, 5, 10, 15],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
}

rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_classif)),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ]
)

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)

print("\nBest Random Forest Score")
print(rf_search.best_score_)

print("\nBest Random Forest Parameters")
print(rf_search.best_params_)

best_rf = rf_search.best_estimator_
rf_predictions = best_rf.predict(X_test)
rf_probabilities = best_rf.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_predictions))

print("\nTuned Random Forest Report")
print(classification_report(y_test, rf_predictions))

print("\nTuned Random Forest ROC AUC")
print(roc_auc_score(y_test, rf_probabilities))

gb_grid = {
    "selector__k": [10, 15, 20, "all"],
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 0.9, 1.0],
}

gb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(score_func=f_classif)),
        ("model", GradientBoostingClassifier(random_state=42)),
    ]
)

gb_search = GridSearchCV(
    estimator=gb_pipeline,
    param_grid=gb_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
)

gb_search.fit(X_train, y_train)

print("\nBest Gradient Boosting Score")
print(gb_search.best_score_)

print("\nBest Gradient Boosting Parameters")
print(gb_search.best_params_)

best_gb = gb_search.best_estimator_
gb_predictions = best_gb.predict(X_test)
gb_probabilities = best_gb.predict_proba(X_test)[:, 1]

print("\nTuned Gradient Boosting Confusion Matrix")
print(confusion_matrix(y_test, gb_predictions))

print("\nTuned Gradient Boosting Report")
print(classification_report(y_test, gb_predictions))

print("\nTuned Gradient Boosting ROC AUC")
print(roc_auc_score(y_test, gb_probabilities))

final_models = {
    "Best Base Model": best_base_model,
    "Tuned Random Forest": best_rf,
    "Tuned Gradient Boosting": best_gb,
}

final_rows = []

for name, model in final_models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    final_rows.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        }
    )

final_df = pd.DataFrame(final_rows)
final_df = final_df.sort_values(by="ROC_AUC", ascending=False)

print("\nFinal Supervised Ranking")
print(final_df)

fpr, tpr, thresholds = roc_curve(y_test, gb_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Gradient Boosting")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

precision, recall, pr_thresholds = precision_recall_curve(y_test, gb_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

threshold_results = []

for threshold in np.arange(0.1, 0.91, 0.05):
    threshold_predictions = (gb_probabilities >= threshold).astype(int)

    threshold_results.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, threshold_predictions),
            "Precision": precision_score(
                y_test,
                threshold_predictions,
                zero_division=0,
            ),
            "Recall": recall_score(y_test, threshold_predictions),
            "F1": f1_score(y_test, threshold_predictions),
        }
    )

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Analysis")
print(threshold_df)

best_threshold = threshold_df.sort_values(by="F1", ascending=False).iloc[0]["Threshold"]

final_predictions = (gb_probabilities >= best_threshold).astype(int)

print("\nBest Threshold")
print(best_threshold)

print("\nFinal Threshold Confusion Matrix")
print(confusion_matrix(y_test, final_predictions))

print("\nFinal Threshold Report")
print(classification_report(y_test, final_predictions))

kmeans_results = []

for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train_processed)
    score = silhouette_score(X_train_processed, labels)

    kmeans_results.append(
        {
            "K": k,
            "Silhouette": score,
        }
    )

kmeans_df = pd.DataFrame(kmeans_results)

print("\nKMeans Results")
print(kmeans_df)

best_k = int(kmeans_df.sort_values(by="Silhouette", ascending=False).iloc[0]["K"])

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_processed)

cluster_profile = pd.DataFrame(X_train_processed, columns=numeric_columns)
cluster_profile["cluster"] = cluster_labels
cluster_profile["target"] = y_train.values

cluster_summary = cluster_profile.groupby("cluster").mean()
cluster_sizes = cluster_profile["cluster"].value_counts().sort_index()

print("\nCluster Sizes")
print(cluster_sizes)

print("\nCluster Summary")
print(cluster_summary)

dbscan = DBSCAN(eps=2.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_train_processed)

print("\nDBSCAN Label Counts")
print(pd.Series(dbscan_labels).value_counts())

isolation_forest = IsolationForest(contamination=0.08, random_state=42)

outlier_labels = isolation_forest.fit_predict(X_train_processed)
outlier_scores = isolation_forest.decision_function(X_train_processed)

outlier_df = X_train.copy()
outlier_df["target"] = y_train.values
outlier_df["outlier_label"] = outlier_labels
outlier_df["outlier_score"] = outlier_scores

print("\nIsolation Forest Outliers")
print(outlier_df["outlier_label"].value_counts())

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.08)

lof_labels = lof.fit_predict(X_train_processed)

print("\nLocal Outlier Factor Labels")
print(pd.Series(lof_labels).value_counts())

feature_importances = best_gb.named_steps["model"].feature_importances_

selected_mask = best_gb.named_steps["selector"].get_support()
selected_columns = X.columns[selected_mask]

importance_df = pd.DataFrame(
    {
        "Feature": selected_columns,
        "Importance": feature_importances,
    }
).sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Gradient Boosting Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

test_results = X_test.copy()
test_results["Actual"] = y_test.values
test_results["Probability"] = gb_probabilities
test_results["Prediction"] = final_predictions

test_results = test_results.sort_values(by="Probability", ascending=False)

print("\nTop Risk Records")
print(test_results.head(20))

comparison_df.to_csv("model_comparison.csv", index=False)
final_df.to_csv("final_model_ranking.csv", index=False)
threshold_df.to_csv("threshold_analysis.csv", index=False)
importance_df.to_csv("feature_importance.csv", index=False)
test_results.to_csv("risk_predictions.csv", index=False)
cluster_summary.to_csv("cluster_summary.csv")

joblib.dump(best_gb, "best_gradient_boosting_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

print("\nSaved Files")
print("model_comparison.csv")
print("final_model_ranking.csv")
print("threshold_analysis.csv")
print("feature_importance.csv")
print("risk_predictions.csv")
print("cluster_summary.csv")
print("best_gradient_boosting_model.pkl")
print("preprocessor.pkl")
