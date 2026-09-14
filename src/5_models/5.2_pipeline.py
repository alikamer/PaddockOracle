################################################
# PIPELINE
################################################

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import joblib
import pandas as pd
from sklearn.model_selection import cross_validate, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import f1_data_prep

warnings.simplefilter(action='ignore', category=Warning)

################################################
# Hyperparameter grid'leri (5.1_research.py ile aynı)
################################################

lr_params = {"C": [0.01, 0.1, 1, 10]}
knn_params = {"n_neighbors": range(2, 30)}
cart_params = {"max_depth": range(1, 15), "min_samples_split": range(2, 20)}
rf_params = {"max_depth": [5, 8, None], "max_features": [5, 7, "sqrt"],
             "min_samples_split": [8, 15, 20], "n_estimators": [100, 200, 300]}
xgboost_params = {"learning_rate": [0.1, 0.01], "max_depth": [3, 5, 8], "n_estimators": [100, 200]}
lightgbm_params = {"learning_rate": [0.01, 0.1], "n_estimators": [200, 300, 500]}
catboost_params = {"iterations": [200, 500], "learning_rate": [0.01, 0.1], "depth": [4, 6]}

classifiers = [('LR', LogisticRegression(max_iter=1000), lr_params),
               ('KNN', KNeighborsClassifier(), knn_params),
               ('CART', DecisionTreeClassifier(random_state=17), cart_params),
               ('RF', RandomForestClassifier(random_state=17), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=17), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state=17, verbose=-1), lightgbm_params),
               ('CatBoost', CatBoostClassifier(random_state=17, verbose=False), catboost_params)]

################################################
# Helper Functions
################################################

def base_models(X, y, cv, scoring="roc_auc"):
    print("Base Models....")
    base_classifiers = [(name, clf) for name, clf, _ in classifiers]
    for name, classifier in base_classifiers:
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

def hyperparameter_optimization(X, y, cv, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X, y)
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

def voting_classifier(best_models, X, y, cv):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('LR', best_models["LR"]),
                                               ('RF', best_models["RF"]),
                                               ('CatBoost', best_models["CatBoost"])],
                                   voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_csv("data/data_processed/results_2010_2025_features.csv")
    X, y = f1_data_prep(df)

    # Final model tüm veriyle (2010-2025) eğitiliyor; train/test ayrımı
    # 5.1_research.py'daki performans doğrulamasına aitti.
    tscv = TimeSeriesSplit(n_splits=5)

    base_models(X, y, cv=tscv)
    best_models = hyperparameter_optimization(X, y, cv=tscv)
    voting_clf = voting_classifier(best_models, X, y, cv=tscv)

    joblib.dump(voting_clf, "models/f1_model.pkl")
    return voting_clf

if __name__ == "__main__":
    print("İşlem başladı")
    main()
