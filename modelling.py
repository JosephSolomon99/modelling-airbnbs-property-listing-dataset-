import os
import json
import joblib
from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer



features, labels = load_airbnb('airbnb-property-listings/tabular_data/clean_tabular_data.csv','Price_Night')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Regression Models
models = {
    'SGDRegressor': SGDRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'BaggingRegressor' : BaggingRegressor(random_state=42),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'XGBRegressor': XGBRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)}

# Define the hyperparameters you want to test for each model
hyperparameters = {
    'SGDRegressor': {"loss":["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],"penalty":["l2", "l1", "elasticnet", None],
        "alpha":[0.0001, 0.001, 0.01, 0.1, 1],"max_iter":[1000,2000,2500, 3000],
        "tol":[0.00001,0.0001, 0.001, 0.01, 0.1],"learning_rate":["constant","optimal","invscaling","adaptive"]},
    'Linear Regression': {},
    'Ridge': {'alpha':[5, 2.5, 4]},
    'BaggingRegressor':{'n_estimators': [10, 50, 100, 150],
        'max_samples': [0.5, 0.7, 0.9, 1.0],
        'max_features': [0.5, 0.7, 0.9, 1.0],
        },
    'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50]},
    'DecisionTreeRegressor': {"max_depth": [None, 5],
        "min_samples_split": [1, 2, 3], "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"], "min_weight_fraction_leaf": [0.0, 0.1],
        "max_leaf_nodes": [None, 5, 10], "min_impurity_decrease": [0.0, 0.1]},
    'XGBRegressor': {'n_estimators': [100, 500], 'max_depth': [5, 7, 9],
            'learning_rate': [0.1, 0.01, 0.001]},
    'Random Forest': {'n_estimators': [100, 500, 1000], 'max_depth': [3, 6, 9],
                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']},
    'Gradient Boosting': {'n_estimators': [100, 500, 1000],'max_depth': [3, 6, 9], 
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],'learning_rate': [0.01, 0.1, 1]}
        }

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def tune_regression_model_hyperparameters(model, name):  
    gridSearchCV = GridSearchCV(estimator=model, param_grid=hyperparameters[name], cv=5, verbose=0, n_jobs=-1, scoring=make_scorer(rmse))
    gridSearchCVResults = gridSearchCV.fit(X_train, y_train)   
    val_rmse = gridSearchCVResults.best_score_
    hyperparams = gridSearchCVResults.best_params_
    model = gridSearchCVResults.best_estimator_
    return model, hyperparams, val_rmse

def performace_metrics(model, val_rmse):
    metrics = {}
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_loss_mse = mean_squared_error(y_train, y_train_pred)
    test_loss_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)     
    metrics["validation_RMSE"] = val_rmse
    metrics["train_loss_MSE"] = train_loss_mse
    metrics["test_loss_MSE"] = test_loss_mse
    metrics["train_R2"] = train_r2
    metrics["test_R2"] = test_r2
    return metrics

def save_model(model, name, hyperparams, metrics, folder='models/regression'):
    # Save Model
    new_folder = os.path.join(folder, name)
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
            model, hyperparams, val_rmse = tune_regression_model_hyperparameters(model, name)
            metrics = performace_metrics(model, val_rmse)
            save_model(model, name, hyperparams, metrics, 'models/regression')

def find_best_model():
    best_model = None
    best_test_loss = float('inf')
    for name, model in models.items():
        model, hyperparams, val_rmse = tune_regression_model_hyperparameters(model, name)
        metrics = performace_metrics(model, val_rmse)
        if metrics['test_loss_MSE'] < best_test_loss:
            best_model = model
            best_test_loss = metrics['test_loss_MSE']
            return best_model

if __name__ == "__main__":
#    evaluate_all_models()
    best_model,hyperparams, metrics = find_best_model()