import os
import joblib
import json
from tabular_data import load_airbnb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,  precision_score, recall_score, f1_score, roc_auc_score



features, labels = load_airbnb('airbnb-property-listings/tabular_data/clean_tabular_data.csv','Category')
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_labels = label_encoder.transform(labels)
labels = encoded_labels
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, multi_class='ovr'),
    'SGDClassifier': SGDClassifier(random_state=42),
    'BaggingClassifier': BaggingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

hyperparameters = {
    'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l2', None], 'solver':['newton-cg', 'sag', 'saga', 'lbfgs']},
    'SGDClassifier': {'alpha': [0.0001, 0.001, 0.01], 'penalty': ['l1', 'l2', 'elasticnet']},
    'BaggingClassifier': {'n_estimators': [10, 50, 100], 'max_samples': [0.5, 1.0], 'max_features': [0.5, 1.0]},
    'Decision Tree': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    'Gradient Boosting': {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    'XGBoost': {'learning_rate': [0.01, 0.1, 1], 'n_estimators': [20, 50, 100, 200], 'max_depth': [0, 1, 2]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']},
    'Gaussian Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
}
def tune_classification_model_hyperparameters(model,name):  
    gridSearchCV = GridSearchCV(estimator=model, param_grid=hyperparameters[name], cv=5, verbose=0, n_jobs=-1)
    gridSearchCVResults = gridSearchCV.fit(X_train, y_train)
    hyperparams = gridSearchCVResults.best_params_
    validation_accuracy = gridSearchCVResults.best_score_
    model = gridSearchCVResults.best_estimator_
    return model, hyperparams, validation_accuracy

def performace_metrics(model, validation_accuracy):
    metrics = {}
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)    
    metrics["Validation_accuracy"] = validation_accuracy
    metrics["Train_accuracy"] = accuracy_score(y_train, y_train_pred)
    metrics["Test_accuracy"] = accuracy_score(y_test, y_test_pred)
    metrics["Precision"] = precision_score(y_test, y_test_pred, average='weighted')
    metrics["Recall"] = recall_score(y_test, y_test_pred, average='weighted')
    metrics["F1 Score"] = f1_score(y_test, y_test_pred, average='weighted')
    return metrics

def save_model(model, name, hyperparams, metrics, task_folder='models/classification'):
    # Save Model
    new_folder = os.path.join(task_folder, name)
    os.makedirs(new_folder, exist_ok=True)
    joblib.dump(model, os.path.join(new_folder, 'model.joblib'))
    # Save Hyperparameters metrics to a JSON file
    with open(os.path.join(new_folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f)
    # Save the performance metrics to a JSON file
    with open(os.path.join(new_folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

def evaluate_all_models():
    for name, model in models.items():
            model, hyperparams, gridSearchCVResults = tune_classification_model_hyperparameters(model, name)
            metrics = performace_metrics(model, gridSearchCVResults)
            save_model(model, name, hyperparams, metrics)

def find_best_model():
    pass

if __name__ == "__main__":
    evaluate_all_models()