from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

features, labels = load_airbnb('airbnb-property-listings/tabular_data/clean_tabular_data.csv','Price_Night')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

model = SGDRegressor()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_loss_mse = mean_squared_error(y_train, y_train_pred)
test_loss_mse = mean_squared_error(y_test, y_test_pred)
train_loss_r2 = r2_score(y_train, y_train_pred)
test_loss_r2 = r2_score(y_test, y_test_pred)

print(f"Train Loss: {train_loss_mse}|" f"Test Loss: {test_loss_mse}")
print(f"Train Loss: {train_loss_r2}|" f"Test Loss: {test_loss_r2}")
