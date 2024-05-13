import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/ApartmentRentPrediction_classification_test.csv")
print(data.isna().sum())
target_Encoder = joblib.load(r'models\classification\dependencies\target_Encoder')

data['RentCategory'] = target_Encoder.transform(data[['RentCategory']])
Y = data['RentCategory']

cityname_mode = joblib.load(r'models\classification\dependencies\cityname_mode.joblib')
data['cityname'].fillna(cityname_mode, inplace=True)

pets_mode = joblib.load(r'models\classification\dependencies\pets_mode.joblib')
data['pets_allowed'].fillna(pets_mode)

amenities_mode = joblib.load(r'models\classification\dependencies\amenities_mode.joblib')
data["amenities"].fillna(amenities_mode, inplace=True)

state_mode = joblib.load(r'models\classification\dependencies\state_mode.joblib')
data["state"].fillna(state_mode, inplace=True)

bathroom_mean = joblib.load(r'models\classification\dependencies\bathroom_mean.joblib')
data["bathrooms"].fillna(bathroom_mean, inplace=True)

bedroom_mean = joblib.load(r'models\classification\dependencies\bedroom_mean.joblib')
data['bedrooms'].fillna(bedroom_mean, inplace=True)

long_mean = joblib.load(r'models\classification\dependencies\long_mean.joblib')
data['longitude'].fillna(long_mean, inplace=True)

lat_mean = joblib.load(r'models\classification\dependencies\lat_mean.joblib')
data['latitude'].fillna(lat_mean, inplace=True)

square_feet_mean = joblib.load(r'models\classification\dependencies\square_feet_mean.joblib')
data['square_feet'].fillna(square_feet_mean, inplace=True)

data['address'] = data.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)

Ordinal_Encoder = joblib.load(r'models\classification\dependencies\classificationEncoder.joblib')
categorical_columns = joblib.load(r'models\classification\dependencies\categorical_columns.joblib')

data[categorical_columns] = Ordinal_Encoder.transform(data[categorical_columns])

significant_numerical_features = joblib.load(r'models\classification\dependencies\significant_numerical_features.joblib')
best_features = joblib.load(r'models\classification\dependencies\best_features_names.joblib')

data = data[list(best_features) + list(significant_numerical_features)]
print(data)
lr_model = joblib.load(r'models\classification\logistic_reg.joblib')
svm_model = joblib.load(r'models\classification\SVM.joblib')
dt_model = joblib.load(r'models\classification\decision_tree.joblib')
rf_model = joblib.load(r'models\classification\random_forest.joblib')
voting_model = joblib.load(r'models\classification\voting.joblib')
stacking_model = joblib.load(r'models\classification\stacking.joblib')

linear_pred = lr_model.predict(data)
print("\nLogistic Regression Predictions were as follow:\n", linear_pred)
print("Logistic Regression Accuracy: ", accuracy_score(Y, linear_pred))

svm_pred = svm_model.predict(data)
print("\nSVM Predictions were as follow:\n", svm_pred)
print("SVM Accuracy: ",accuracy_score(Y, svm_pred))

dt_pred = dt_model.predict(data)
print("\nDecision Tree Predictions were as follow:\n", dt_pred)
print("Decision Tree Accuracy: ",accuracy_score(Y, dt_pred))

rf_pred = rf_model.predict(data) 
print("\RF Predictions were as follow:\n", rf_pred)
print("Random Forest Accuracy: ",accuracy_score(Y, rf_pred))

voting_pred = voting_model.predict(data)
print("\nVoting Predictions were as follow:\n",voting_pred)
print("Ensemble Voting Accuracy: ",accuracy_score(Y, voting_pred))

stacking_pred = stacking_model.predict(data) 
print("\nStacking Predictions were as follow:\n",stacking_pred)
print("Ensemble Stacking Accuracy: ",accuracy_score(Y, stacking_pred))
