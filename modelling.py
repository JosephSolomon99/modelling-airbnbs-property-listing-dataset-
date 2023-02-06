from tabular_data import load_airbnb
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

features, labels = load_airbnb('airbnb-property-listings/tabular_data/clean_tabular_data.csv','Price_Night')

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

model = SGDRegressor()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

print(f"Train Loss: {train_loss}|" f"Test Loss: {test_loss}")
