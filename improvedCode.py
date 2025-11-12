# ==========================================
# HEART DISEASE CLASSIFICATION PIPELINE WITH PCHF
# ==========================================

# libraries for data transformation
import pandas as pd
import numpy as np
import warnings

# libraries for preprocessing and tuning
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# libraries for machine learning models that we are testing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# result metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# for PCHF-like feature engineering
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(r"C:\Users\fiona\OneDrive\Desktop\DS340Wproject\dataset.csv")

# transform target variable into binary format - 0 and 1
df['num'] = (df['num'] > 0).astype(int)

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
)  # 0.1765 ≈ 15% of original dataset

# preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder())
        ]), categorical_cols)
    ]
)

# ADDITION: PCHF FEATURE ENGINEERING METHOD
def pchf(X, y=None, explained_variance_threshold=0.95, model_based_selection=True, top_n=8, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for dimensionality reduction
    pca_full = PCA(random_state=random_state)
    X_pca_full = pca_full.fit_transform(X_scaled)

    # Find how many components are needed for the variance threshold
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= explained_variance_threshold) + 1
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    print(f"[PCHF] PCA retained {n_components} components explaining {cum_var[n_components-1]*100:.2f}% of variance.")

    # Model-based selection
    selected_features = list(range(n_components))
    if model_based_selection and y is not None and n_components > top_n:
        rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
        rf.fit(X_pca, y)
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        X_pca = X_pca[:, sorted_idx]
        selected_features = sorted_idx

        print(f"[PCHF] Selected top {top_n} components via RandomForest_")

    return X_pca, pca, selected_features


# models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# parameter grids
param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        "solver": ["liblinear", "lbfgs"],
        "penalty": ["l2"]
    },
    "KNN": {
        "n_neighbors": list(range(1, 31)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    },
    "SVM": {
        "C": [0.1, 0.5, 1, 2, 5, 10, 50],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 150, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "max_depth": [3, 4, 5, 6, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0]
    }
}

#train and evaluate
def train_and_evaluate(models, param_grids, X_train, y_train, X_val, y_val):
    results = []

    for name, model in models.items():
        print(f"\n=== {name} ===")

        # base preprocessing + pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # baseline (no tuning)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        base_acc = accuracy_score(y_val, y_pred)
        base_f1 = f1_score(y_val, y_pred)

        print(f"Baseline:  Accuracy = {base_acc:.4f} | F1 = {base_f1:.4f}")

        X_train_prep = preprocessor.fit_transform(X_train)
        X_val_prep = preprocessor.transform(X_val)

        X_train_pchf, pca_model, selected_feats = pchf(
            X_train_prep, y_train,
            explained_variance_threshold=0.97,
            model_based_selection=True,
            top_n=10
        )
        X_val_pchf = pca_model.transform(X_val_prep)[:, selected_feats]

        # refit with tuned hyperparameters using PCA features
        grid = GridSearchCV(model, param_grid=param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_pchf, y_train)

        y_pred_grid = grid.predict(X_val_pchf)
        tuned_acc = accuracy_score(y_val, y_pred_grid)
        tuned_f1 = f1_score(y_val, y_pred_grid)

        print(f"Tuned (with PCHF): Accuracy = {tuned_acc:.4f} | F1 = {tuned_f1:.4f}")
        print(f"Best Params: {grid.best_params_}")
        results.append({
            "Model": name,
            "Baseline Accuracy": base_acc,
            "Tuned Accuracy": tuned_acc,
            "Best Params": grid.best_params_
        })

    return pd.DataFrame(results)

# run everything
results = train_and_evaluate(models, param_grids, X_train, y_train, X_val, y_val)
print("\n=== Summary of All Models (with PCHF) ===")
print(results)

