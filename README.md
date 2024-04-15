# Apartment Rent Prediction Model

## Overview
This repository contains a machine learning model designed to predict apartment rental prices based on various features. The model utilizes a dataset with information on apartment listings including features such as number of bathrooms, bedrooms, amenities, location, and more.

## Dataset
The dataset used for training and testing the model consists of the following features:

- **id**: Unique identifier for each apartment listing.
- **category**: Category of the listing.
- **title**: Title of the listing.
- **body**: Description of the listing.
- **amenities**: Amenities available in the apartment.
- **bathrooms**: Number of bathrooms in the apartment.
- **bedrooms**: Number of bedrooms in the apartment.
- **currency**: Currency used for the price.
- **fee**: Any additional fees associated with the rental.
- **has_photo**: Boolean indicating whether the listing has photos.
- **pets_allowed**: Boolean indicating whether pets are allowed in the apartment.
- **price**: Rental price.
- **price_display**: Display of the rental price.
- **price_type**: Type of the rental price.
- **square_feet**: Size of the apartment in square feet.
- **address**: Address of the apartment.
- **cityname**: City where the apartment is located.
- **state**: State where the apartment is located.
- **latitude**: Latitude coordinate of the apartment location.
- **longitude**: Longitude coordinate of the apartment location.
- **source**: Source of the listing.
- **time**: Timestamp of the listing.

## Usage
1. **Data Preprocessing**: Before using the model, preprocess the dataset to handle missing values, encode categorical variables, and scale numerical features.
2. **Model Training**: Train the model using suitable algorithms such as linear regression, decision trees, or neural networks. Tune hyperparameters as necessary.
3. **Model Evaluation**: Evaluate the trained model's performance using appropriate metrics such as mean absolute error, mean squared error, or R-squared.
4. **Prediction**: Use the trained model to predict rental prices for new apartment listings based on their features.

## Dependencies
Ensure you have the following dependencies installed:
- Python (version 3.x)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualization)
