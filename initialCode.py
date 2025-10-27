import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv(r"C:\Users\fiona\OneDrive\Desktop\DS340Wproject\dataset.csv")

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'dataset']
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

imputer = SimpleImputer(strategy="median")
for col in ['trestbps', 'ca', 'oldpeak', 'chol', 'thalch']:
    df[col] = imputer.fit_transform(df[[col]])

df = df[df['trestbps'] != 0]

df['dataset'] = LabelEncoder().fit_transform(df['dataset'].astype(str))

X = df.drop('num', axis=1)
y = df['num']

for col in categorical_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=8 random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_features = pd.DataFrame(
    X_pca,
    columns=[f"PCA_Feature_{i+1}" for i in range(X_pca.shape[1])]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("Data loaded and preprocessed. Shape:", X_train.shape)

# Define models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('KNeighbors Classifier', KNeighborsClassifier()),
    ('Support Vector Machine', SVC(random_state=42)),
    ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('XGBoost', XGBClassifier(random_state=42))
]

best_model = None
best_accuracy = 0.0
from sklearn.pipeline import Pipeline
# Iterate over the models and evaluate their performance
for name, model in models:
    pipeline = Pipeline([
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    mean_accuracy = scores.mean()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}")
    print(f"Cross Validation Accuracy: {mean_accuracy}")
    print(f"Test Accuracy: {accuracy}\n")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

print("Best Model:", best_model)

# Hyperparameter tuning
def hyperparameter_tuning(X, y, categorical_columns, models):
    results = {}
    X_encoded = X.copy()
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        X_encoded[col] = label_encoder.fit_transform(X_encoded[col])

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        param_grid = {}
        if model_name == 'Logistic Regression':
            param_grid = {'C': [0.1, 1, 10]}
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7]}
        elif model_name == 'Gaussian Naive Bayes':
            param_grid = {'var_smoothing': np.logspace(-9, 0, 10)}
        elif model_name == 'SVM':
            param_grid = {'C': [0.1, 1], 'kernel': ['linear']}
        elif model_name == 'Decision Tree':
            param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
        elif model_name == 'Random Forest':
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        elif model_name == 'AdaBoost':
            param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
        elif model_name == 'Gradient Boosting':
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5]}
        elif model_name == 'XGBoost':
            param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
        else:
            continue

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {'best_params': best_params, 'accuracy': accuracy}

    return results

# Define models for tuning
models_for_tuning = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier()
}

# Perform hyperparameter tuning
results = hyperparameter_tuning(X, y, categorical_cols, models_for_tuning)
print(results)