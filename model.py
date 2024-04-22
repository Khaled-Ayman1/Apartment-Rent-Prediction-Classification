
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

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

print("Most common value in pets_allowed:", data["pets_allowed"].mode()[0])
print("Most common value in amenities:", data["amenities"].mode()[0])

values_to_choose_from = data['address'].dropna().unique()  # Get unique values from the column, excluding NaNs
random_values = np.random.choice(values_to_choose_from, size=len(data), replace=True)  # Generate random values

# Fill the column with random values
data['address'] = random_values


print("Most common value in bedrooms:", data["bedrooms"].mode()[0])
print("Most common value in bathrooms:", data["bathrooms"].mode()[0])
print("Most common value in cityname:", data["cityname"].mode()[0])
print("Most common value in state:", data["state"].mode()[0])
print("Most common value in latitude:", data["latitude"].mode()[0])
print("Most common value in longitude:", data["longitude"].mode()[0])

# Handling missing values
data["pets_allowed"].fillna('Cats,Dogs', inplace=True)
data["amenities"].fillna('Parking', inplace=True)
data["bathrooms"].fillna('1.0', inplace=True)
data["bedrooms"].fillna('1.0', inplace=True)
data["cityname"].fillna('Austin', inplace=True)
data["state"].fillna('Austin', inplace=True)
data["pets_allowed"].fillna('TX,Dogs', inplace=True)
data["latitude"].fillna('30.3054', inplace=True)
data["longitude"].fillna('-97.7497', inplace=True)

print("Checking for Missing Values after handling:")
print(data.isna().sum())

print("Information about the DataFrame:")
print(data.info())

# Preprocessing price_display column
data['price_display'] = data['price_display'].str.replace('[^\d.]', '', regex=True).astype(float)



columns_to_drop = ['category','id','price', 'title', 'body', 'source', 'time','currency', 'fee','has_photo','price_type']
data = data.drop(columns=columns_to_drop)

data.info()

# Encoding
def encode_categorical(data, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

categorical_columns = ['amenities', 'cityname', 'state', 'pets_allowed', 'address']
data_encoded = encode_categorical(data, categorical_columns)

# Displaying the encoded DataFrame
print("Encoded DataFrame:")
print(data.head())

sns.pairplot(data, y_vars=['price_display'], x_vars=data.columns, height=2)
#plt.show()

columns_with_outliers = ['amenities', 'bathrooms', 'bedrooms', 'pets_allowed', 'price_display', 'square_feet', 'latitude', 'longitude']

# Calculate Z-scores for each column
data[columns_with_outliers] = data[columns_with_outliers].astype(float)  # Convert to float
z_scores = np.abs(stats.zscore(data[columns_with_outliers]))

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

sns.pairplot(data_cleaned, y_vars=['price_display'], x_vars=data.columns, height=2)
#plt.show()

# print("Mean House Rent:", round(data["price_display"].mean()))
# print("Median House Rent:", round(data["price_display"].median()))
# print("Highest House Rent:", round(data["price_display"].max()))
# print("Lowest House Rent:", round(data["price_display"].min()))
# print("\n")
# print("Highest House Rent after removing the outliers:", round(data_cleaned["price_display"].max()))
# print("Lowest House Rent after removing the outliers:", round(data_cleaned["price_display"].min()))

#correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
#plt.show()

#Get the correlation between the features
corr = data.corr()
#Top Features
top_features = corr.index[abs(corr['price_display'])>0.18]
#top_features Correlation plot
top_corr = data[top_features].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()

Y = data['price_display']
top_features = top_features.drop('price_display')
f = ['bathrooms', 'bedrooms', 'square_feet', 'cityname', 'longitude', 'latitude']
X = data[f]


# data splitting
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

# model training
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

# model training
# linear model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(x_train, y_train)

# model testing
y_train_prediction = linear_reg.predict(x_train)
y_predict = linear_reg.predict(x_test)

print("\nlinear model")
print('Mean Square Error for testing', metrics.mean_squared_error(y_test, y_predict))
print('Mean Square Error for training', metrics.mean_squared_error(y_train, y_train_prediction))


