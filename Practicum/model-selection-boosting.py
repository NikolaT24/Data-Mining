import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ]),
    "Random Forest": RandomForestClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

model_results = []

for model_name, model in models.items():
    scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=10,
        scoring="accuracy"
    )

    model_results.append({
        "Model": model_name,
        "Mean Accuracy": scores.mean(),
        "Standard Deviation": scores.std()
    })

results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values(by="Mean Accuracy", ascending=False)

print("Model Comparison")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nBest Model:", best_model_name)
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

adaboost_pipeline = Pipeline([
    ("classifier", AdaBoostClassifier(
        estimator=DecisionTreeClassifier(random_state=0),
        random_state=0
    ))
])

adaboost_params = {
    "classifier__n_estimators": [50, 100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    "classifier__estimator__max_depth": [1, 2, 3]
}

adaboost_grid = GridSearchCV(
    estimator=adaboost_pipeline,
    param_grid=adaboost_params,
    scoring="accuracy",
    cv=10,
    n_jobs=-1
)

adaboost_grid.fit(X_train, y_train)

print("\nBest AdaBoost Accuracy")
print(adaboost_grid.best_score_)

print("\nBest AdaBoost Parameters")
print(adaboost_grid.best_params_)

best_adaboost = adaboost_grid.best_estimator_

y_pred_adaboost = best_adaboost.predict(X_test)

print("\nAdaBoost Test Results")
print(confusion_matrix(y_test, y_pred_adaboost))
print(accuracy_score(y_test, y_pred_adaboost))
print(classification_report(y_test, y_pred_adaboost))

gradient_boosting_params = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.8, 0.9, 1.0]
}

gradient_boosting_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=0),
    param_grid=gradient_boosting_params,
    scoring="accuracy",
    cv=10,
    n_jobs=-1
)

gradient_boosting_grid.fit(X_train, y_train)

print("\nBest Gradient Boosting Accuracy")
print(gradient_boosting_grid.best_score_)

print("\nBest Gradient Boosting Parameters")
print(gradient_boosting_grid.best_params_)

best_gradient_boosting = gradient_boosting_grid.best_estimator_

y_pred_gradient_boosting = best_gradient_boosting.predict(X_test)

print("\nGradient Boosting Test Results")
print(confusion_matrix(y_test, y_pred_gradient_boosting))
print(accuracy_score(y_test, y_pred_gradient_boosting))
print(classification_report(y_test, y_pred_gradient_boosting))

random_forest_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

random_forest_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_distributions=random_forest_params,
    n_iter=20,
    scoring="accuracy",
    cv=10,
    random_state=0,
    n_jobs=-1
)

random_forest_random_search.fit(X_train, y_train)

print("\nBest Random Forest Accuracy")
print(random_forest_random_search.best_score_)

print("\nBest Random Forest Parameters")
print(random_forest_random_search.best_params_)

best_random_forest = random_forest_random_search.best_estimator_

y_pred_random_forest = best_random_forest.predict(X_test)

print("\nRandom Forest Test Results")
print(confusion_matrix(y_test, y_pred_random_forest))
print(accuracy_score(y_test, y_pred_random_forest))
print(classification_report(y_test, y_pred_random_forest))

final_models = {
    "Best Basic Model": best_model,
    "Best AdaBoost": best_adaboost,
    "Best Gradient Boosting": best_gradient_boosting,
    "Best Random Forest": best_random_forest
}

final_results = []

for model_name, model in final_models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    final_results.append({
        "Model": model_name,
        "Test Accuracy": accuracy_score(y_test, predictions)
    })

final_results_df = pd.DataFrame(final_results)
final_results_df = final_results_df.sort_values(by="Test Accuracy", ascending=False)

print("\nFinal Model Ranking")
print(final_results_df)
