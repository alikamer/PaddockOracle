################################################
# RESEARCH
################################################

# 1. Veriyi okuma
# 2. Modele girecek sütunların seçimi
# 3. Kategorik sütunun sayıya çevrilmesi
# 4. Kronolojik ayırma (train / test)
# 5. Base Models
# 6. Hyperparameter Optimization
# 7. Feature Importance
# 8. Voting Classifier (Ortak Karar)

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Veriyi okuma
################################################

df = pd.read_csv("data/data_processed/results_2010_2025_features.csv")

df.shape
df.head()
df.columns

################################################
# 2. Modele hangi feature'ları sokacağız? Bazıların drop etmemiz gerekli, data leakage riski taşıyanlar var + gereksizler
################################################

# Bu sütunlar yarış bittikten sonra öğrenilen bilgileri taşır, bırakırsak model ezber yapar.
leak_cols = ["position", "position_text", "points", "laps", "status"]

# Kimlik sütunları -  işlevsizler, modelin görmesi gereksiz.
id_cols = ["driver_id", "constructor_id", "circuit_id", "race_name", "date"]

# Sezon numarası bir özellik değil, sadece ayırma için lazım
drop_cols = leak_cols + id_cols + ["season", "is_podium"]

y = df["is_podium"]
X = df.drop(columns=drop_cols)

X.shape
X.columns
X.head()

################################################
# 3. Kategorik sütunların sayıya çevrilme kısmı (unified_teams için)
################################################

# utils.py
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

X["team_unified"].value_counts()

X = one_hot_encoder(X, ["team_unified"], drop_first=True)

X.shape
X.head()

################################################
# 4. Kronolojik ayırma (train / test)
################################################

# HOLDOUT yaptığımız kısım
train_mask = df["season"] <= 2023
test_mask = df["season"] >= 2024

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

X_train.shape
X_test.shape


################################################
# 5. Base Models
################################################

def base_models(X, y, cv, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(max_iter=1000)),
                   ('KNN', KNeighborsClassifier()),
                   ('CART', DecisionTreeClassifier(random_state=17)),
                   ('RF', RandomForestClassifier(random_state=17)),
                   ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=17)),
                   ('LightGBM', LGBMClassifier(random_state=17, verbose=-1)),
                   ('CatBoost', CatBoostClassifier(random_state=17, verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")

tscv = TimeSeriesSplit(n_splits=5)  #cv yerine tscv bizim veri setimiz için elzem, kronolojik olarak bölüp eğitmek gerekiyor.
base_models(X_train, y_train, cv=tscv)

'''
┌──────────┬─────────┐
│  model   │ roc_auc │
├──────────┼─────────┤
│ RF       │ 0.9224  │
├──────────┼─────────┤
│ LR       │ 0.9205  │
├──────────┼─────────┤
│ LightGBM │ 0.9182  │
├──────────┼─────────┤
│ XGBoost  │ 0.9123  │
├──────────┼─────────┤
│ KNN      │ 0.8490  │
├──────────┼─────────┤
│ CART     │ 0.7283  │
└──────────┴─────────┘
'''

################################################
# 6. Hyperparameter Optimization
################################################

lr_params = {"C": [0.01, 0.1, 1, 10]}

knn_params = {"n_neighbors": range(2, 30)}

cart_params = {"max_depth": range(1, 15),
               "min_samples_split": range(2, 20)}

rf_params = {"max_depth": [5, 8, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [100, 200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                   "max_depth": [3, 5, 8],
                   "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                    "n_estimators": [200, 300, 500]}

catboost_params = {"iterations": [200, 500],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6]}

classifiers = [('LR', LogisticRegression(max_iter=1000), lr_params),
               ('KNN', KNeighborsClassifier(), knn_params),
               ('CART', DecisionTreeClassifier(random_state=17), cart_params),
               ('RF', RandomForestClassifier(random_state=17), rf_params),
               ('XGBoost', XGBClassifier(eval_metric='logloss', random_state=17), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state=17, verbose=-1), lightgbm_params),
               ('CatBoost', CatBoostClassifier(random_state=17, verbose=False), catboost_params)]

def hyperparameter_optimization(X, y, cv, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X, y)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X_train, y_train, cv=tscv)

################################################
# 7. Feature Importance
################################################

def plot_importance(model, features, num=len(X_train), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(best_models["CART"], X_train)
plot_importance(best_models["RF"], X_train)
plot_importance(best_models["XGBoost"], X_train)
plot_importance(best_models["LightGBM"], X_train)
plot_importance(best_models["CatBoost"], X_train)

################################################
# 8. Voting Classifier (Ortak Karar)
################################################

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

voting_clf = voting_classifier(best_models, X_train, y_train, cv=tscv)
