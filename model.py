import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_classif


data = pd.read_csv("data/ApartmentRentPrediction.csv")

# Creating Dataframe
print("Shape of the DataFrame:", data.shape)

data.head()

print("Statistical description of the DataFrame:")
print(data.describe())

print("Columns in the DataFrame:")
print(data.columns)

# Preprocessing and Feature Selection
print("Checking for Missing Values:")
print(data.isna().sum())

print("Checking for Duplicated Data:")
print(data.duplicated().sum())

# Fill missing address with city and state name
data['address'] = data.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)

bedroom_mode = data["bedrooms"].mode()[0]
bathroom_mode = data["bathrooms"].mode()[0]
cityname_mode = data["cityname"].mode()[0]
state_mode = data["state"].mode()[0]
lat_mode = data["latitude"].mode()[0]
long_mode = data["longitude"].mode()[0]
pets_mode = data["pets_allowed"].mode()[0]
amenities_mode = data["amenities"].mode()[0]

print("Most common value in bedrooms:", bedroom_mode)
print("Most common value in bathrooms:", bathroom_mode)
print("Most common value in cityname:", cityname_mode)
print("Most common value in state:", state_mode)
print("Most common value in latitude:", lat_mode)
print("Most common value in longitude:", long_mode)

# Handling missing values

data["amenities"].fillna(amenities_mode, inplace=True)
data["pets_allowed"].fillna(pets_mode, inplace=True)
data["bathrooms"].fillna(bathroom_mode, inplace=True)
data["bedrooms"].fillna(bedroom_mode, inplace=True)
data["cityname"].fillna(cityname_mode, inplace=True)
data["state"].fillna(state_mode, inplace=True)
data["latitude"].fillna(lat_mode, inplace=True)
data["longitude"].fillna(long_mode, inplace=True)

print("Checking for Missing Values after handling:")
print(data.isna().sum())

print("Information about the DataFrame:")
print(data.info())

# Preprocessing price_display column
data['price_display'] = data['price_display'].str.replace('[^\d.]', '', regex=True).astype(float)

print(data['price_display'].describe())

columns_to_drop = ['category', 'id', 'price', 'title', 'body', 'source', 'time', 'currency', 'fee']
data = data.drop(columns=columns_to_drop)

# Save categorical columns before dropping
categorical_columns = ['amenities', 'cityname', 'state', 'address', 'price_type', 'pets_allowed', 'has_photo']
data_categorical = data[categorical_columns]

data.info()

# Encoding
def encode_categorical(data, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

data = encode_categorical(data, categorical_columns)

# Displaying the encoded DataFrame
print("Encoded DataFrame:")
print(data.head())

for col in data.columns:
    sns.pairplot(data, y_vars=['price_display'], x_vars=col, height=2)
    #plt.show()

# Calculate Z-scores for each column
data = data.astype(float)  # Convert to float
z_scores = np.abs(stats.zscore(data))

# Set threshold for identifying outliers (e.g., Z-score > 3)
threshold = 3

# Find indices of outliers
outlier_indices = np.where(z_scores > threshold)

# Remove outliers from DataFrame
data_cleaned = data.drop(outlier_indices[0])

data = data_cleaned.apply(pd.to_numeric, errors='coerce').dropna()

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

for col in data.columns:
    sns.pairplot(data, y_vars=['price_display'], x_vars=col, height=2)
    #plt.show()

print(data['price_display'].describe())

# correlation matrix
corr = data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Encode categorical features before ANOVA
data_encoded = encode_categorical(data, categorical_columns)

# ANOVA for categorical features
anova_results = f_classif(data_encoded[categorical_columns], data['price_display'])
anova_p_values = pd.Series(anova_results[1], index=categorical_columns)

# Select significant categorical features based on p-value threshold
significant_categorical_features = anova_p_values[anova_p_values < 0.05].index.tolist()

print("Significant categorical features based on ANOVA p-values:", significant_categorical_features)


# Top Features
top_features = corr.index[abs(corr['price_display']) > 0.1]
print(top_features)

# top_features Correlation plot
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

Y = data['price_display']
top_features = top_features.drop('price_display')
print(top_features)
X = data_encoded[list(significant_categorical_features) + list(top_features)]
print(X)

# data splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

# polynomial model
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(x_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# model testing
prediction = poly_model.predict(poly_features.fit_transform(x_test))
prediction1 = poly_model.predict(poly_features.fit_transform(x_train))

print("polynomial model")
print('Mean Square Error for testing', metrics.mean_squared_error(y_test, prediction))
print('Mean Square Error for training', metrics.mean_squared_error(y_train, prediction1))
print("r2 score:", r2_score(y_test, prediction))

# linear model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train, y_train)

# model testing
y_train_prediction = linear_reg.predict(x_train)
y_predict = linear_reg.predict(x_test)

print("\nlinear model")
print('Mean Square Error for testing', metrics.mean_squared_error(y_test, y_predict))
print('Mean Square Error for training', metrics.mean_squared_error(y_train, y_train_prediction))
print("r2 score:", r2_score(y_test, y_predict))

# poly model with cross validation
print('\ncross validation')
model_1_poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
X_train_poly_model_1 = model_1_poly_features.fit_transform(x_train)

# fit the transformed features to Linear Regression
poly_model1 = linear_model.LinearRegression()
scores = cross_val_score(poly_model1, X_train_poly_model_1, y_train, scoring='neg_mean_squared_error', cv=8)
model_1_score = abs(scores.mean())

poly_model1.fit(X_train_poly_model_1, y_train)
print("model 1 cross validation score is " + str(model_1_score))

model_2_poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly_model_2 = model_2_poly_features.fit_transform(x_train)

# fit the transformed features to Linear Regression
poly_model2 = linear_model.LinearRegression()
scores = cross_val_score(poly_model2, X_train_poly_model_2, y_train, scoring='neg_mean_squared_error', cv=8)
model_2_score = abs(scores.mean())
poly_model2.fit(X_train_poly_model_2, y_train)

print("model 2 cross validation score is " + str(model_2_score))

# predicting on test data-set
prediction = poly_model1.predict(model_1_poly_features.fit_transform(x_test))
print('\nModel 1 Test Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print("r2 score:", r2_score(y_test, prediction))

# predicting on test data-set
prediction = poly_model2.predict(model_2_poly_features.fit_transform(x_test))
print('Model 2 Test Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print("r2 score:", r2_score(y_test, prediction))



