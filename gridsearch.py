## Grid Search from Scratch

# def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
#     keys, values = zip(*hyperparameters.items())
#     yield from (dict(zip(keys, v)) for v in itertools.product(*values))

# def custom_tune_regression_model_hyperparameters(model_class,X_train,y_train,X_validation,y_validation,X_test,y_test, hyperparameters):
#     best_model = None
#     best_hyperparams = {}
#     best_metrics = {}
#     for hyperparams in grid_search(hyperparameters):
#         model = model_class(**hyperparams)       
#         model.fit(X_train, y_train)
        
#         y_validation_pred = model.predict(X_validation)             
#         validation_RMSE = mean_squared_error(y_validation, y_validation_pred, squared=False)

#         if not best_model or validation_RMSE < best_metrics["validation_RMSE"]:
#             best_model = model
#             best_metrics["validation_RMSE"] = validation_RMSE
#             best_hyperparams = hyperparams

#     y_train_pred = best_model.predict(X_train)
#     y_test_pred = best_model.predict(X_test)
#     train_loss_mse = mean_squared_error(y_train, y_train_pred)
#     test_loss_mse = mean_squared_error(y_test, y_test_pred)
#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)
    
#     best_metrics["train_loss_MSE"] = train_loss_mse
#     best_metrics["test_loss_MSE"] = test_loss_mse
#     best_metrics["train_R2"] = train_r2
#     best_metrics["test_R2"] = test_r2
    

#     return best_model, best_hyperparams, best_metrics

# best_model, best_hyperparams, best_metrics = custom_tune_regression_model_hyperparameters(model_class,X_train,y_train,X_validation,y_validation,X_test,y_test,hyperparameters)

