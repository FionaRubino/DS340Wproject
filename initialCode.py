# libraries for data transformation
import pandas as pd
import numpy as np

#libraries for preprocessing and tuning
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#libraries for machine learning models that we are testing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # for the future
from sklearn.tree import DecisionTreeClassifier # for the future 
from xgboost import XGBClassifier

# result metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# for PCHF-like feature engineering (non parent paper)
from sklearn.decomposition import PCA # for the future

# load dataset
df = pd.read_csv(r"C:\Users\fiona\OneDrive\Desktop\DS340Wproject\dataset.csv")

# transform target variable into binary format - 0 and 1
df['num'] = (df['num'] > 0).astype(int) # multiple values in data besides 0 and 1

# separate features and target
X = df.drop('num', axis=1)
y = df['num']

# categorical and numerical columns
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'dataset']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# split dataset: 70% train, 15% validation, 15% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, stratify=y_train_full, random_state=42
)  # 0.1765 ≈ 15% of original dataset - gets validation set

# preprocessing: 
preprocessor = ColumnTransformer(
    transformers=[
        # handles missing numerical values since models can't handle them
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),

        # handles categorical values with most frequent, then OneHotEncode
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
        ]), categorical_cols)
    ]
)

# models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

#parameter grids
param_grids = {
    "Logistic Regression": {
        "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        "model__solver": ["liblinear", "lbfgs"],  # try another solve
        "model__penalty": ["l2"]
    },
    "KNN": {
        "model__n_neighbors": list(range(1, 31)),  # up to 30 neighbors
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "minkowski"]  # try different distances
    },
    "SVM": {
        "model__C": [0.1, 0.5, 1, 2, 5, 10, 50],
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1]
    },
    "XGBoost": {
        "model__n_estimators": [50, 100, 150, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0]
    }
}

# train and test everything at once
def train_and_evaluate(models, param_grids, X_train, y_train, X_val, y_val):
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")

        # pipeline for repetition
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # training no gridsearch
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        base_acc = accuracy_score(y_val, y_pred)
        base_f1 = f1_score(y_val, y_pred)

        print(f"Baseline:  Accuracy = {base_acc:.4f} | F1 = {base_f1:.4f}")

        #gridsearch
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid=param_grids[name], cv=cv_strategy, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        # re-evaluate on the same validation split 
        y_pred_grid = grid.predict(X_val)
        tuned_acc = accuracy_score(y_val, y_pred_grid)
        tuned_f1 = f1_score(y_val, y_pred_grid)

        print(f"Tuned:     Accuracy = {tuned_acc:.4f} | F1 = {tuned_f1:.4f}")
        print(f"Best Params: {grid.best_params_}")

        results.append({
            "Model": name,
            "Baseline Accuracy": base_acc,
            "Tuned Accuracy": tuned_acc,
            "Best Params": grid.best_params_
        })

    return pd.DataFrame(results)


#Run everything
results = train_and_evaluate(models, param_grids, X_train, y_train, X_val, y_val)
print("\nSummary")
print(results)