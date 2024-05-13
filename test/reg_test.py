import pandas as pd
import joblib
from sklearn import metrics
from sklearn.metrics import r2_score


data = pd.read_csv("data/ApartmentRentPrediction_test.csv")
data['price_display'] = data['price_display'].str.replace(r'[^\d.]', '', regex=True).astype(float)
Y = data['price_display']


cityname_mode = joblib.load(r'models\regression\dependencies\cityname_mode.joblib')
data['cityname'].fillna(cityname_mode, inplace=True)

pets_mode = joblib.load(r'models\regression\dependencies\pets_mode.joblib')
data['pets_allowed'].fillna(pets_mode)

amenities_mode = joblib.load(r'models\regression\dependencies\amenities_mode.joblib')
data["amenities"].fillna(amenities_mode, inplace=True)

state_mode = joblib.load(r'models\regression\dependencies\state_mode.joblib')
data["state"].fillna(state_mode, inplace=True)

bathroom_mean = joblib.load(r'models\regression\dependencies\bathroom_mean.joblib')
data["bathrooms"].fillna(bathroom_mean, inplace=True)

bedroom_mean = joblib.load(r'models\regression\dependencies\bedroom_mean.joblib')
data['bedrooms'].fillna(bedroom_mean, inplace=True)

long_mean = joblib.load(r'models\regression\dependencies\long_mean.joblib')
data['longitude'].fillna(long_mean, inplace=True)

square_feet_mean = joblib.load(r'models\regression\dependencies\square_feet_mean.joblib')
data['square_feet'].fillna(square_feet_mean, inplace=True)

data['address'] = data.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)
Ordinal_loaded = joblib.load(r'models\regression\dependencies\ordinal_Encoder.joblib')

categorical_columns = joblib.load(r'models\regression\dependencies\categorical_columns.joblib')
data[categorical_columns] = Ordinal_loaded.transform(data[categorical_columns])

significant_categorical_features = joblib.load(r'models\regression\dependencies\significant_categorical_features.joblib')

print(significant_categorical_features)

top_features = joblib.load(r'models\regression\dependencies\top_features.joblib')
data = data[list(significant_categorical_features) + list(top_features)]

print(data)

poly_loaded, model_loaded = joblib.load(r'models\regression\cv_poly_model.joblib')
data_poly = poly_loaded.transform(data)
Y_pred = model_loaded.predict(data_poly)
print("\nPolynomial Predictions were as follow:\n",Y_pred)
print("\nPolynomial Model MSE is " , metrics.mean_squared_error(Y, Y_pred))
print("r2 score:", r2_score(Y, Y_pred))



linear_model = joblib.load(r"models\regression\cv_lin_model.joblib")
pred = linear_model.predict(data)
print("\nLinear Predictions were as follow:\n",pred)

print("\nLinear Model MSE is " , metrics.mean_squared_error(Y, pred))
print("r2 score:", r2_score(Y, pred))





