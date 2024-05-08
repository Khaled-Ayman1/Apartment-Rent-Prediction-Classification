import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_classif
import joblib

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data/ApartmentRentPrediction.csv")
data['price_display'] = data['price_display'].str.replace(r'[^\d.]', '', regex=True).astype(float)


columns_to_drop = ['price_display','category', 'id', 'price', 'title', 'body', 'source', 'time', 'currency', 'fee']
X = data
X = X.drop(columns=columns_to_drop)
Y = data['price_display']
# X.head()



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)



#print(X_train.isna().sum())


bedroom_mean = X_train['bedrooms'].mean()
bathroom_mean = X_train['bathrooms'].mean()
cityname_mode = X_train['cityname'].mode()[0]
state_mode = X_train['state'].mode()[0]
lat_mean = X_train['latitude'].mean()
long_mean = X_train['longitude'].mean()
pets_mode = X_train['pets_allowed'].mode()[0]
amenities_mode = X_train["amenities"].mode()[0]


# Fill missing address with city and state name
X_train['address'] = X_train.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)

X_train["amenities"].fillna(amenities_mode, inplace=True)
X_train['bathrooms'].fillna(bathroom_mean, inplace=True)
X_train['bedrooms'].fillna(bedroom_mean, inplace=True)
X_train['pets_allowed'].fillna(pets_mode, inplace=True)
X_train['cityname'].fillna(cityname_mode, inplace=True)
X_train['state'].fillna(state_mode, inplace=True)
X_train['latitude'].fillna(lat_mean, inplace=True)
X_train['longitude'].fillna(long_mean, inplace=True)



#print(X_train.isna().sum())


ordinal_Encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

joblib.dump(ordinal_Encoder, "models/ordinal_Encoder.joblib")


categorical_columns = ['amenities', 'cityname', 'state', 'address', 'price_type', 'pets_allowed', 'has_photo']
X_train[categorical_columns] = ordinal_Encoder.fit_transform(X_train[categorical_columns])
# X_train.head()

train_Data = pd.concat([X_train, Y_train], axis=1)
# train_Data.head()


train_Data = train_Data.astype(float)
z_scores = np.abs(stats.zscore(train_Data))

threshold = 3

# Find indices of outliers
outlier_indices = np.where(z_scores > threshold)[0]

# Remove outliers from DataFrame using iloc to safely select index positions
data_cleaned = train_Data.iloc[~train_Data.index.isin(outlier_indices)]

# Ensure all data is numeric and drop rows with NaNs
train_Data = data_cleaned.apply(pd.to_numeric, errors='coerce').dropna()



Q1 = train_Data.quantile(0.25)
Q3 = train_Data.quantile(0.75)
IQR = Q3 - Q1
train_Data = train_Data[~((train_Data < (Q1 - 1.5 * IQR)) | (train_Data > (Q3 + 1.5 * IQR))).any(axis=1)]


data_categorical = train_Data[categorical_columns]

# ANOVA for categorical features
anova_results = f_classif(train_Data[categorical_columns], train_Data['price_display'])
anova_p_values = pd.Series(anova_results[1], index=categorical_columns)

significant_categorical_features = anova_p_values[anova_p_values < 0.05].index.tolist()

#print("Significant categorical features based on ANOVA p-values:", significant_categorical_features)




corr = train_Data.corr()

# plt.figure(figsize=(12, 8))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()



top_features = corr.index[abs(corr['price_display']) > 0.1]
#print(top_features)
Y_train = train_Data["price_display"]
top_features = top_features.drop('price_display')

X_train = train_Data[list(significant_categorical_features) + list(top_features)]


X_test["amenities"].fillna(amenities_mode, inplace=True)
X_test['bathrooms'].fillna(bathroom_mean, inplace=True)
X_test['bedrooms'].fillna(bathroom_mean, inplace=True)
X_test['state'].fillna(state_mode, inplace=True)
X_test['longitude'].fillna(long_mean, inplace=True)
X_test['latitude'].fillna(lat_mean, inplace=True)
X_test['address'] = X_test.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)


X_test[categorical_columns] = ordinal_Encoder.transform(X_test[categorical_columns])
    
X_test = X_test[list(significant_categorical_features) + list(top_features)]
#print(X_test.isna().sum())



# polynomial model
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)

joblib.dump(poly_model, "models/poly_model.joblib")


# model testing
prediction = poly_model.predict(poly_features.fit_transform(X_test))
prediction1 = poly_model.predict(poly_features.fit_transform(X_train))

print("Polynomial Model")
print('Mean Square Error for testing', metrics.mean_squared_error(Y_test, prediction))
print('Mean Square Error for training', metrics.mean_squared_error(Y_train, prediction1))
print("r2 score:", r2_score(Y_test, prediction))

# linear model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, Y_train)

joblib.dump(linear_reg, "models/linear_reg.joblib")


# model testing
y_train_prediction = linear_reg.predict(X_train)
y_predict = linear_reg.predict(X_test)

print("\nLinear Model")
print('Mean Square Error for testing', metrics.mean_squared_error(Y_test, y_predict))
print('Mean Square Error for training', metrics.mean_squared_error(Y_train, y_train_prediction))
print("r2 score:", r2_score(Y_test, y_predict))

# poly model with cross validation
print('\nPolynomial with Cross Validation')
model_1_poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model1 = linear_model.LinearRegression()
scores = cross_val_score(poly_model1, X_train_poly_model_1, Y_train, scoring='neg_mean_squared_error', cv=8)
model_1_score = abs(scores.mean())

poly_model1.fit(X_train_poly_model_1, Y_train)
print("Polynomial Model cross validation score is " + str(model_1_score))

joblib.dump(poly_model1, "models/cv_poly_model.joblib")


# fit the transformed features to Linear Regression
cv_lin_reg = linear_model.LinearRegression()
scores = cross_val_score(cv_lin_reg, X_train, Y_train, scoring='neg_mean_squared_error', cv=8)
modelscore = abs(scores.mean())

cv_lin_reg.fit(X_train, Y_train)
print("Linear Model cross validation score is " + str(modelscore))

joblib.dump(cv_lin_reg, "models/cv_linear_reg.joblib")


# predicting on test data-set
prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
print('\nCV Polymodel Test Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
print("r2 score:", r2_score(Y_test, prediction))

# predicting on test data-set
prediction = cv_lin_reg.predict(X_test)
print('CV Linear Test Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
print("r2 score:", r2_score(Y_test, prediction))