from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import typing
import numpy as np


features, labels = load_airbnb('airbnb-property-listings/tabular_data/clean_tabular_data.csv','Price_Night')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.2, random_state=42)
model_class = SGDRegressor

hyperparameters = {"loss":["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "penalty":["l2", "l1", "elasticnet", None],
        "alpha":[0.0001, 0.001, 0.01, 0.1, 1],
        "max_iter":[1000,1500, 2000, 2500],
        "tol":[0.00001,0.0001, 0.001, 0.01, 0.1],
        "learning_rate":["constant","optimal","invscaling","adaptive"]}

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

def custom_tune_regression_model_hyperparameters(model_class,X_train,y_train,X_validation,y_validation,X_test,y_test, hyperparameters):
    best_model = None
    best_hyperparams = {}
    best_metrics = {}
    for hyperparams in grid_search(hyperparameters):
        model = model_class(**hyperparams)
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)
        
        model.fit(X_train, y_train)
        
        y_validation_pred = model.predict(X_validation)     
        
        validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)

        if not best_model or validation_RMSE < best_metrics["validation_RMSE"]:
            best_model = model
            best_metrics["validation_RMSE"] = validation_RMSE
            best_hyperparams = hyperparams

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    train_loss_mse = mean_squared_error(y_train, y_train_pred)
    test_loss_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    best_metrics["train_loss_MSE"] = train_loss_mse
    best_metrics["test_loss_MSE"] = test_loss_mse
    best_metrics["train_R2"] = train_r2
    best_metrics["test_R2"] = test_r2
    

    return best_model, best_hyperparams, best_metrics

best_model, best_hyperparams, best_metrics = custom_tune_regression_model_hyperparameters(model_class,X_train,y_train,X_validation,y_validation,X_test,y_test,hyperparameters)
print(best_model)
print(best_hyperparams)
print(best_metrics)
