import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_classif
import joblib
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest




data = pd.read_csv("data/ApartmentRentClassification.csv")


columns_to_drop = ['category', 'id', 'title', 'body', 'source', 'time', 'currency', 'fee', 'price_type', 'RentCategory']
X = data
X = X.drop(columns=columns_to_drop)
X.head()


ordinalE = OrdinalEncoder()
data['RentCategory'] = ordinalE.fit_transform(data[['RentCategory']])
Y = data['RentCategory']
Y


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)



print(X_train.isna().sum())

bedroom_mean = X_train['bedrooms'].mean()
bathroom_mean = X_train['bathrooms'].mean()
cityname_mode = X_train['cityname'].mode()[0]
state_mode = X_train['state'].mode()[0]
lat_mean = X_train['latitude'].mean()
long_mean = X_train['longitude'].mean()
pets_mode = X_train['pets_allowed'].mode()[0]
amenities_mode = X_train["amenities"].mode()[0]




X_train["amenities"].fillna(amenities_mode, inplace=True)
X_train['bathrooms'].fillna(bathroom_mean, inplace=True)
X_train['bedrooms'].fillna(bedroom_mean, inplace=True)
X_train['pets_allowed'].fillna(pets_mode, inplace=True)
X_train['cityname'].fillna(cityname_mode, inplace=True)
X_train['state'].fillna(state_mode, inplace=True)
X_train['latitude'].fillna(lat_mean, inplace=True)
X_train['longitude'].fillna(long_mean, inplace=True)


# Fill missing address with city and state name
X_train['address'] = X_train.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)



print(X_train.isna().sum())



ordinal_Encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

#joblib.dump(ordinal_Encoder, 'Ordinal_encoder2.joblib')

categorical_columns = ['amenities', 'cityname', 'state', 'address', 'pets_allowed', 'has_photo']
X_train[categorical_columns] = ordinal_Encoder.fit_transform(X_train[categorical_columns])
X_train.head()

train_Data = pd.concat([X_train, Y_train], axis=1)



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


numerical_columns = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
# ANOVA for categorical features
anova_results = f_classif(train_Data[numerical_columns], train_Data['RentCategory'])
anova_p_values = pd.Series(anova_results[1], index=numerical_columns)

significant_numerical_features = anova_p_values[anova_p_values < 0.05].index.tolist()

print("Significant numerical features based on ANOVA p-values:", significant_numerical_features)



corr = train_Data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

top_features = corr.index[abs(corr['RentCategory']) > 0.1]
print(top_features)



cate = train_Data[categorical_columns]
# Apply Chi-squared test
chi2_selector = SelectKBest(chi2, k=4)
chi2_selector.fit(cate, train_Data['RentCategory'])

# Get the scores for each feature
chi_scores = pd.DataFrame({
    'Feature': categorical_columns,
    'Score': chi2_selector.scores_
}).sort_values(by='Score', ascending=False)

print(chi_scores)

best_features_indices = chi2_selector.get_support(indices=True)
best_features_names = [cate.columns[i] for i in best_features_indices]

best_features_names



Y_train = train_Data['RentCategory']
X_train = train_Data[list(best_features_names) + list(significant_numerical_features)]
X_train.head()


print(X_test.isna().sum())
X_test["amenities"].fillna(amenities_mode, inplace=True)
X_test['bathrooms'].fillna(bathroom_mean, inplace=True)
X_test['bedrooms'].fillna(bedroom_mean, inplace=True)
X_test['pets_allowed'].fillna(pets_mode, inplace=True)
X_test['cityname'].fillna(cityname_mode, inplace=True)
X_test['state'].fillna(state_mode, inplace=True)
X_test['latitude'].fillna(lat_mean, inplace=True)
X_test['longitude'].fillna(long_mean, inplace=True)


# Fill missing address with city and state name
X_test['address'] = X_test.apply(lambda row: f"{row['cityname']}, {row['state']}" if pd.isnull(row['address']) else row['address'], axis=1)


print(X_test.isna().sum())

X_test[categorical_columns] = ordinal_Encoder.transform(X_test[categorical_columns])

X_test = X_test[list(best_features_names) + list(significant_numerical_features)]


