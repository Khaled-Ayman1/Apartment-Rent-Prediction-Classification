import pandas as pd
import numpy as np
import time
import joblib
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data/ApartmentRentClassification.csv")

columns_to_drop = ['category', 'id', 'title', 'body', 'source', 'time', 'currency', 'fee', 'price_type', 'RentCategory']
X = data
X = X.drop(columns=columns_to_drop)

# X.head()

ordinalE = OrdinalEncoder()
data['RentCategory'] = ordinalE.fit_transform(data[['RentCategory']])
Y = data['RentCategory']

joblib.dump(ordinalE, "models/classification/dependencies/target_Encoder")

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
square_feet_mean = X_train['square_feet'].mode()[0]

joblib.dump(bedroom_mean, 'models/classification/dependencies/bedroom_mean.joblib')
joblib.dump(bathroom_mean, 'models/classification/dependencies/bathroom_mean.joblib')
joblib.dump(cityname_mode, 'models/classification/dependencies/cityname_mode.joblib')
joblib.dump(state_mode, 'models/classification/dependencies/state_mode.joblib')
joblib.dump(lat_mean, 'models/classification/dependencies/lat_mean.joblib')
joblib.dump(long_mean, 'models/classification/dependencies/long_mean.joblib')
joblib.dump(pets_mode, 'models/classification/dependencies/pets_mode.joblib')
joblib.dump(amenities_mode, 'models/classification/dependencies/amenities_mode.joblib')
joblib.dump(square_feet_mean, 'models/classification/dependencies/square_feet_mean.joblib')

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

categorical_columns = ['amenities', 'cityname', 'state', 'address', 'pets_allowed', 'has_photo']

joblib.dump(categorical_columns, 'models/classification/dependencies/categorical_columns.joblib')

X_train[categorical_columns] = ordinal_Encoder.fit_transform(X_train[categorical_columns])
# X_train.head()

joblib.dump(ordinal_Encoder, 'models/classification/dependencies/classificationEncoder.joblib')


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

# ANOVA for numerical features
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

joblib.dump(best_features_names, 'models/classification/dependencies/best_features_names.joblib')
joblib.dump(significant_numerical_features, 'models/classification/dependencies/significant_numerical_features.joblib')

# print(X_train.head())

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
lr_model = LogisticRegression(multi_class='multinomial', solver="lbfgs", max_iter=100000)

start_timer = time.time()

lr_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression 1 Train: ", timer) 

start_timer = time.time()

lr_predictions = lr_model.predict(X_test)
end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression 1 Test: ", timer)
lr_accuracy = accuracy_score(Y_test, lr_predictions)
print("\nMultinomial Logistic Regression Accuracy (100k iterations & lbfgs):", lr_accuracy)

 
lr1 = LogisticRegression(multi_class='multinomial',  solver='saga', max_iter=100000)
start_timer = time.time()
lr1.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nLogistic Regression 2 Train: ", timer) 

start_timer = time.time()

y_pred = lr1.predict(X_test)
end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression 2 Test: ", timer) 
accuracylr1 = accuracy_score(Y_test, y_pred)
print("\nAccuracy (100k iterations & saga):", accuracylr1)


lr2 = LogisticRegression(multi_class='multinomial',  solver='newton-cg', max_iter=100000)
start_timer = time.time()

lr2.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nLogistic Regression 3 Train: ", timer) 

start_timer = time.time()

y_pred = lr2.predict(X_test)
end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression Model 3 Test: ",timer) 

accuracylr2 = accuracy_score(Y_test, y_pred)
print("\nAccuracy (100k iterations & newton-cg):", accuracylr2)

start_timer = time.time()

lr3 = LogisticRegression(multi_class='multinomial',  solver='newton-cg', max_iter=1000)
lr3.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nLogistic Regression 4 Train: ", timer) 

start_timer = time.time()

best_lr_predictions = lr3.predict(X_test)
end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression Model 4 Test: ", timer) 
best_lr_accuracy = accuracy_score(Y_test, best_lr_predictions)
print("\nAccuracy (1k iterations & newton-cg):", best_lr_accuracy)

joblib.dump(lr3, 'models/classification/logistic_reg.joblib')

lr5 = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000000)
start_timer = time.time()
lr5.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nLogistic Regression Model 5 Train: ", timer) 

start_timer = time.time()

y_pred = lr5.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("Logistic Regression 5 Test: ", timer)

accuracylr5 = accuracy_score(Y_test, y_pred)
print("\nAccuracy (1M iterations & newton-cg):",accuracylr5)


lr4 = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=5000000)

start_timer = time.time()

lr4.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nLogistic Regression 6 Train: ", timer) 

start_timer = time.time()

y_pred = lr4.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer  
print("Logistic Regression 6 Test: ", timer) 

accuracylr4 = accuracy_score(Y_test, y_pred)
print("\nAccuracy (5M iterations & newton-cg):",accuracylr4)



#output
"""
Accuracy (100k iterations & saga): 0.5538888888888889
Accuracy (100k iterations & newton-cg): 0.5833333333333334
Accuracy (1k iterations & newton-cg): 0.5833333333333334
Accuracy (1M iterations & newton-cg): 0.5833333333333334
Accuracy (5M iterations & newton-cg): 0.5833333333333334

"""

# Random Forest Classifier
rf_model = RandomForestClassifier()

start_timer = time.time()

rf_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nRandom Forest Train: ", timer) 

start_timer = time.time()
rf_predictions = rf_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("Random Forest Test: ", timer) 

rf_accuracy = accuracy_score(Y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)


#ouput
# Random Forest Accuracy: 0.758

# Hyperparameter tuning for Random Forest Classifier


rf_params = {
    'n_estimators': [100,5000],
    'max_depth': [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'n_jobs': [-1]
}

rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)

start_timer = time.time()

rf_grid.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nRandom Forest Grid Search + Train: ", timer) 

best_rf_model = rf_grid.best_estimator_

start_timer = time.time()

best_rf_predictions = best_rf_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer
print("Random Forest Grid Search + Test: ",timer) 

best_rf_accuracy = accuracy_score(Y_test, best_rf_predictions)
print("\nBest Random Forest Accuracy:", best_rf_accuracy)
print("Best Random Forest Parameters:", rf_grid.best_params_)

joblib.dump(best_rf_model, 'models/classification/random_forest.joblib')

# output
# Best Random Forest Accuracy: 0.7666666666666667
# Best Random Forest Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}

#Support Vector Machine Classifier
svm_model = SVC()

start_timer = time.time()

svm_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nSVM Train Time: ", timer) 

start_timer = time.time()

svm_predictions = svm_model.predict(X_test)

end_timer = time.time()
timer = start_timer - end_timer
print("SVM Test Time",timer) 

svm_accuracy = accuracy_score(Y_test, svm_predictions)
print("\nSupport Vector Machine Accuracy:", svm_accuracy)



# Hyperparameter tuning for SVM with RBF kernel

svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'],
}
svm_grid = GridSearchCV(SVC(), svm_params, cv=5)

start_timer = time.time()

svm_grid.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nSVM Grid Search + Train Time: ", timer) 


rbf_svm_model = svm_grid.best_estimator_

start_timer = time.time()

rbf_svm_predictions = rbf_svm_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("SVM Grid Search + Train Time: ", timer) 

rbf_svm_accuracy = accuracy_score(Y_test, rbf_svm_predictions)
print("\nSupport Vector Machine Accuracy (RBF Kernel):", rbf_svm_accuracy)
print("Support Vector Machine Parameters (RBF Kernel):", svm_grid.best_params_)

joblib.dump(rbf_svm_model, 'models/classification/SVM.joblib')



#output
# Support Vector Machine Accuracy (RBF Kernel): 0.5316666666666666
# Support Vector Machine Parameters (RBF Kernel): {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}

best_svm_predictions=svm_predictions

# Decision Tree Classifier

dt_model = DecisionTreeClassifier()

start_timer = time.time()

dt_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nDT Train Time: ", timer) 

start_timer = time.time()

dt_predictions = dt_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("DT Test Time: ", timer) 

dt_accuracy = accuracy_score(Y_test, dt_predictions)
print("\nDecision Tree Accuracy:", dt_accuracy)





#Hyperparameter tuning for Decision Tree Classifier
dt_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)

start_timer = time.time()

dt_grid.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nDT GS + Train: ", timer) 

best_dt_model = dt_grid.best_estimator_

start_timer = time.time()

best_dt_predictions = best_dt_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("DT GS + Test: ", timer) 

joblib.dump(best_dt_model, 'models/classification/decision_tree.joblib')


best_dt_accuracy = accuracy_score(Y_test, best_dt_predictions)
print("\nBest Decision Tree Accuracy:", best_dt_accuracy)
print("Best Decision Tree Parameters:", dt_grid.best_params_)


#output
# Best Decision Tree Accuracy: 0.6561111111111111
# Best Decision Tree Parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5}

# Print classification reports for all models
print("\nLogistic Regression Classification Report:")
print(classification_report(Y_test, best_lr_predictions ))

print("\nRandom Forest Classification Report:")
print(classification_report(Y_test, best_rf_predictions))

print("\nSupport Vector Machine Classification Report:")
print(classification_report(Y_test, best_svm_predictions))

# Print classification report for Decision Tree Classifier
print("\nDecision Tree Classification Report:")
print(classification_report(Y_test, best_dt_predictions))

# Ensemble using Voting Classifier

ensemble_model = VotingClassifier(estimators=[
    ('Random Forest', best_rf_model),
    ('Logistic Regression', lr_model),
    ('Support Vector Machine', svm_model),
    ('Decision Tree', dt_model)
])

start_timer = time.time()

ensemble_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nEnsemble Voting Train: ", timer) 

start_timer = time.time()

ensemble_predictions = ensemble_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("Ensemble Voting Test: ", timer) 

ensemble_accuracy = accuracy_score(Y_test, ensemble_predictions)
print("\nEnsemble Voting Classifier Accuracy:", ensemble_accuracy)
print("Ensemble Voting Classifier Classification Report:")
print(classification_report(Y_test, ensemble_predictions))


joblib.dump(ensemble_model, 'models/classification/voting.joblib')


# Ensemble using stacking
#Define the base models and meta-model:
# Base models
base_models = [
    ('Random Forest', best_rf_model),
    ('Support Vector Machine', svm_model),
    ('Decision Tree', dt_model)
]
meta_model = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

start_timer = time.time()

stacking_model.fit(X_train, Y_train)

end_timer = time.time()
timer = end_timer - start_timer 
print("\nEnsemble Stacking Train ", timer) 

start_timer = time.time()

stacking_predictions = stacking_model.predict(X_test)

end_timer = time.time()
timer = end_timer - start_timer 
print("Ensemble Stacking Test: ", timer) 

start_timer = time.time()

stacking_accuracy = accuracy_score(Y_test, stacking_predictions)
print("\nStacking Classifier Accuracy:", stacking_accuracy)
print("Stacking Classifier Classification Report:")
print(classification_report(Y_test, stacking_predictions))

joblib.dump(stacking_model, 'models/classification/stacking.joblib')


# Trained Output
"""
Logistic Regression 1 Train:  3.0676891803741455
Logistic Regression 1 Test:  0.0010006427764892578
Multinomial Logistic Regression Accuracy (100k iterations & lbfgs): 0.5744444444444444
Logistic Regression 2 Train:  1.3603057861328125
Logistic Regression 2 Test:  0.001010894775390625    
Accuracy (100k iterations & saga): 0.5538888888888889
Logistic Regression 3 Train:  1.0462350845336914
Logistic Regression Model 3 Test:  0.0010006427764892578  
Accuracy (100k iterations & newton-cg): 0.5833333333333334
Logistic Regression 4 Train:  1.0342319011688232
Logistic Regression Model 4 Test:  0.0010004043579101562
Accuracy (1k iterations & newton-cg): 0.5833333333333334
Logistic Regression Model 5 Train:  1.0472376346588135
Logistic Regression 5 Test:  0.0010113716125488281      
Accuracy (1M iterations & newton-cg): 0.5833333333333334
Logistic Regression 6 Train:  1.0542271137237549
Logistic Regression 6 Test:  0.0009996891021728516      
Accuracy (5M iterations & newton-cg): 0.5833333333333334
Random Forest Train:  0.8111820220947266
Random Forest Test:  0.029008865356445312 
Random Forest Accuracy: 0.7544444444444445
Random Forest Grid Search + Train:  292.28937554359436
Random Forest Grid Search + Test:  -6.972500801086426
Best Random Forest Accuracy: 0.7677777777777778
Best Random Forest Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 5000, 'n_jobs': -1}
SVM Train Time:  0.7361652851104736
SVM Test Time -0.6271421909332275
Support Vector Machine Accuracy: 0.5277777777777778
SVM Grid Search + Train Time:  39.916977405548096
SVM Grid Search + Train Time:  1.3613057136535645
Support Vector Machine Accuracy (RBF Kernel): 0.5316666666666666
Support Vector Machine Parameters (RBF Kernel): {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
DT Train Time:  0.03300666809082031
DT Test Time:  0.0019998550415039062
Decision Tree Accuracy: 0.6088888888888889
DT GS + Train:  2.455552339553833
DT GS + Test:  0.0020003318786621094
Best Decision Tree Accuracy: 0.6661111111111111
Best Decision Tree Parameters: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
Logistic Regression Classification Report:
              precision    recall  f1-score   support

         0.0       0.73      0.33      0.46       317
         1.0       0.49      0.50      0.50       533
         2.0       0.61      0.71      0.66       950

    accuracy                           0.58      1800
   macro avg       0.61      0.52      0.54      1800
weighted avg       0.60      0.58      0.57      1800

Random Forest Classification Report:
              precision    recall  f1-score   support

         0.0       0.87      0.72      0.79       317
         1.0       0.74      0.68      0.71       533
         2.0       0.75      0.83      0.79       950

    accuracy                           0.77      1800
   macro avg       0.79      0.74      0.76      1800
weighted avg       0.77      0.77      0.77      1800

Support Vector Machine Classification Report:
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00       317
         1.0       0.00      0.00      0.00       533
         2.0       0.53      1.00      0.69       950

    accuracy                           0.53      1800
   macro avg       0.18      0.33      0.23      1800
weighted avg       0.28      0.53      0.36      1800

Decision Tree Classification Report:
              precision    recall  f1-score   support

         0.0       0.63      0.68      0.65       317
         1.0       0.63      0.62      0.63       533
         2.0       0.70      0.68      0.69       950

    accuracy                           0.67      1800
   macro avg       0.65      0.66      0.66      1800
weighted avg       0.67      0.67      0.67      1800

Ensemble Voting Train:  16.861063718795776
Ensemble Voting Test:  2.722611904144287
Ensemble Voting Classifier Accuracy: 0.7377777777777778
Ensemble Voting Classifier Classification Report:
              precision    recall  f1-score   support

         0.0       0.85      0.65      0.74       317
         1.0       0.72      0.63      0.67       533
         2.0       0.72      0.83      0.77       950

    accuracy                           0.74      1800
   macro avg       0.76      0.70      0.73      1800
weighted avg       0.74      0.74      0.74      1800

Ensemble Stacking Train  74.38628673553467
Ensemble Stacking Test:  3.569803476333618
Stacking Classifier Accuracy: 0.7616666666666667     
Stacking Classifier Classification Report:
              precision    recall  f1-score   support

         0.0       0.86      0.69      0.77       317
         1.0       0.75      0.67      0.70       533
         2.0       0.74      0.84      0.79       950

    accuracy                           0.76      1800
   macro avg       0.78      0.73      0.75      1800
weighted avg       0.77      0.76      0.76      1800

"""