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

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier

data = pd.read_csv("/content/ApartmentRentPrediction_Milestone2.csv")

columns_to_drop = ['category', 'id', 'title', 'body', 'source', 'time', 'currency', 'fee', 'price_type', 'RentCategory']
X = data
X = X.drop(columns=columns_to_drop)

# X.head()

ordinalE = OrdinalEncoder()
data['RentCategory'] = ordinalE.fit_transform(data[['RentCategory']])
Y = data['RentCategory']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

# print(X_train.isna().sum())

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


# print(X_train.isna().sum())


ordinal_Encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

#joblib.dump(ordinal_Encoder, 'Ordinal_encoder2.joblib')

categorical_columns = ['amenities', 'cityname', 'state', 'address', 'pets_allowed', 'has_photo']
X_train[categorical_columns] = ordinal_Encoder.fit_transform(X_train[categorical_columns])
# X_train.head()

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

# print("Significant numerical features based on ANOVA p-values:", significant_numerical_features)

cate = train_Data[categorical_columns]

# Apply Chi-squared test
chi2_selector = SelectKBest(chi2, k=4)
chi2_selector.fit(cate, train_Data['RentCategory'])

# Get the scores for each feature
chi_scores = pd.DataFrame({
    'Feature': categorical_columns,
    'Score': chi2_selector.scores_
}).sort_values(by='Score', ascending=False)

# print(chi_scores)

best_features_indices = chi2_selector.get_support(indices=True)
best_features_names = [cate.columns[i] for i in best_features_indices]


Y_train = train_Data['RentCategory']
X_train = train_Data[list(best_features_names) + list(significant_numerical_features)]
# X_train.head()


# print(X_test.isna().sum())

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

# print(X_test.isna().sum())

X_test[categorical_columns] = ordinal_Encoder.transform(X_test[categorical_columns])

X_test = X_test[list(best_features_names) + list(significant_numerical_features)]

# Multinomial Logistic Regression Classifier
lr_model = LogisticRegression(multi_class='multinomial', max_iter=100000)
lr_model.fit(X_train, Y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(Y_test, lr_predictions)
print("Multinomial Logistic Regression Accuracy:", lr_accuracy)

lr1 = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000000)
lr1.fit(X_train, Y_train)
y_pred = lr1.predict(X_test)
accuracylr1 = accuracy_score(Y_test, y_pred)
print("Accuracy lr1:",accuracylr1)

lr2 = LogisticRegression(multi_class='multinomial',  solver='saga', max_iter=100000)
lr2.fit(X_train, Y_train)
y_pred = lr2.predict(X_test)
accuracylr2 = accuracy_score(Y_test, y_pred)
print("Accuracy lr2:", accuracylr2)

lr3 = LogisticRegression(multi_class='multinomial',  solver='newton-cg', max_iter=1000)
lr3.fit(X_train, Y_train)
best_lr_predictions = lr3.predict(X_test)
best_lr_accuracy = accuracy_score(Y_test, best_lr_predictions)
print("Accuracy lr3:", best_lr_accuracy)
#output
# Multinomial Logistic Regression Accuracy: 0.5761111111111111
# Accuracy lr1: 0.5761111111111111
# Accuracy lr2: 0.5538888888888889
# Accuracy lr3: 0.5833333333333334

# Random Forest Classifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(Y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
#ouput
# Random Forest Accuracy: 0.7522222222222222

# Hyperparameter tuning for Random Forest Classifier
rf_params = {
    'n_estimators': [100,5000],
    'max_depth': [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
rf_grid.fit(X_train, Y_train)
best_rf_model = rf_grid.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_accuracy = accuracy_score(Y_test, best_rf_predictions)
print("Best Random Forest Accuracy:", best_rf_accuracy)
print("Best Random Forest Parameters:", rf_grid.best_params_)
#output
# Best Random Forest Accuracy: 0.7666666666666667
# Best Random Forest Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}

# Support Vector Machine Classifier
svm_model = SVC()
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(Y_test, svm_predictions)
print("Support Vector Machine Accuracy:", svm_accuracy)

# Hyperparameter tuning for SVM with RBF kernel
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(), svm_params, cv=5)
svm_grid.fit(X_train, Y_train)
rbf_svm_model = svm_grid.best_estimator_
rbf_svm_predictions = rbf_svm_model.predict(X_test)
rbf_svm_accuracy = accuracy_score(Y_test, rbf_svm_predictions)
print("Support Vector Machine Accuracy (RBF Kernel):", rbf_svm_accuracy)
print("Support Vector Machine Parameters (RBF Kernel):", svm_grid.best_params_)
#output
#  Support Vector Machine Accuracy (RBF Kernel): 0.5316666666666666
# Support Vector Machine Parameters (RBF Kernel): {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}

 best_svm_predictions=svm_predictions

# Decision Tree Classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(Y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)

# Hyperparameter tuning for Decision Tree Classifier
dt_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
dt_grid.fit(X_train, Y_train)
best_dt_model = dt_grid.best_estimator_
best_dt_predictions = best_dt_model.predict(X_test)
best_dt_accuracy = accuracy_score(Y_test, best_dt_predictions)
print("Best Decision Tree Accuracy:", best_dt_accuracy)
print("Best Decision Tree Parameters:", dt_grid.best_params_)

#output
# Best Decision Tree Accuracy: 0.6561111111111111
# Best Decision Tree Parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}

# Print classification reports for all models
print("Logistic Regression Classification Report:")
print(classification_report(Y_test, best_lr_predictions ))

print("Random Forest Classification Report:")
print(classification_report(Y_test, best_rf_predictions))

print("Support Vector Machine Classification Report:")
print(classification_report(Y_test, best_svm_predictions))

# Print classification report for Decision Tree Classifier
print("Decision Tree Classification Report:")
print(classification_report(Y_test, best_dt_predictions))

# Ensemble using Voting Classifier
from sklearn.ensemble import VotingClassifier
ensemble_model = VotingClassifier(estimators=[
    ('Random Forest', best_rf_model),
    ('Logistic Regression', lr_model),
    ('Support Vector Machine', svm_model),
    ('Decision Tree', dt_model)
])
ensemble_model.fit(X_train, Y_train)
ensemble_predictions = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print("Ensemble Voting Classifier Accuracy:", ensemble_accuracy)
print("Ensemble Voting Classifier Classification Report:")
print(classification_report(Y_test, ensemble_predictions))

# Ensemble using stacking
#Define the base models and meta-model:
# Base models
base_models = [
    ('Random Forest', best_rf_model),
    ('Logistic Regression', lr_model),
    ('Support Vector Machine', svm_model),
    ('Decision Tree', dt_model)
]
meta_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, Y_train)
stacking_predictions = stacking_model.predict(X_test)
stacking_accuracy = accuracy_score(Y_test, stacking_predictions)
print("Stacking Classifier Accuracy:", stacking_accuracy)
print("Stacking Classifier Classification Report:")
print(classification_report(Y_test, stacking_predictions))