import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import style
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import pickle

pokemon_data = pd.read_csv('pokemon_alopez247.csv')

#Exploring the dataset

print(pokemon_data.describe())
print(pokemon_data.isnull().sum())
numeric_data = pokemon_data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
# correlation_matrix.to_csv("correlation_matrix.csv", index=False)


# plt.scatter(pokemon_data['HP'],pokemon_data['Attack'])
# plt.xlabel("HP")
# plt.ylabel("Attack")
#

# z_scores = stats.zscore(pokemon_data)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# pokemon_data12 = pokemon_data[filtered_entries]

mean_value = np.mean(pokemon_data['Attack'])
std_deviation = np.std(pokemon_data['Attack'])



# Select only numeric columns for z-score normalization
numeric_columns = pokemon_data.select_dtypes(include=['number']).columns
pokemon_data1 = pokemon_data[numeric_columns]

# Calculate z-scores for numeric columns
z_scores = stats.zscore(pokemon_data1)

# Create a DataFrame with z-scores
z_scores_df = pd.DataFrame(z_scores, columns=numeric_columns)

# Combine the z-scores with non-numeric columns
your_dataframe_no_outliers = pd.concat([z_scores_df, pokemon_data.select_dtypes(exclude=['number'])], axis=1)

# Selecting relevant features and target variable
# #
# print("this is")
# print(your_dataframe_no_outliers)
# #
X = your_dataframe_no_outliers[['HP', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Height_m','Weight_kg','Catch_Rate']]
y = your_dataframe_no_outliers['Attack']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
import pickle
pickle.dump(scaler,open('scaling.pkl','wb'))

# Create a linear regression model
model = LinearRegression()
ridge_model = Ridge(alpha=0.1)


# Fit the model on the training data
model.fit(X_train, y_train)
ridge_model.fit(X_train,y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# print(y_pred)
# y_pred = scaler.inverse_transform(y_pred.reshape(1,-1))
# print(y_pred)

# pre = pd.DataFrame(y_test)
# pre.to_csv("test.csv")
# # print(y_pred.shape())
# print(y_test)

y_pred1 = ridge_model.predict(X_test)
# Evaluate the model
mae = np.mean(np.abs(y_test - y_pred1))
mse = mean_squared_error(y_test, y_pred1)
r2 = r2_score(y_test, y_pred1)

pickle.dump(ridge_model,open('regmodel.pkl','wb'))

trained_model = pickle.load(open('regmodel.pkl','rb'))

prediction = trained_model.predict(scaler.transform(X_train[0].reshape(1,-1)))
print(prediction)
prediction = trained_model.predict(X_train[0].reshape(1,-1))



prediction = prediction * std_deviation + mean_value


print(prediction)
print(f"Mean Squared Error: {mse:.2f}")
print(y_pred.shape)
print(y_test.shape)
# prediction = scaler.inverse_transform(prediction.reshape(1,-1))




threshold = 0.1  # 5% threshold
accurate_predictions = (abs(y_pred - y_test) / y_test) < threshold
accuracy_percentage = (accurate_predictions.sum() / len(y_test)) * 100

# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"R-squared (R2) Score: {r2:.2f}")
#
# error_percentage = (np.mean(np.abs(y_test))/mae)
# print(f"Error Percentage: {error_percentage:.2f}")
# print(f"Accuracy percentage:{accuracy_percentage:.2f}")

############################################################
# Create a Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # Train the model
# rf_model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred_rf = rf_model.predict(X_test)
#
#
#
# # Evaluate the model performance
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# print(f'MAE Random Forest: {mae_rf}')
# accuracy_percentage = (accurate_predictions.sum() / len(y_test)) * 100
# print(f"Accuracy percentage:{accuracy_percentage:.2f}")
# error_percentage = (np.mean(np.abs(y_test))/mae)
# print(f"Error Percentage: {error_percentage:.2f}")









########################################################
# alpha_values = [0.1, 1.0, 5.0, 10.0]
#
# for alpha in alpha_values:
#     print(alpha)
#     # Create a Ridge Regression model with the specified alpha
#     ridge_model1 = Ridge(alpha=alpha)
#
#     # Train the Ridge model
#     ridge_model1.fit(X_train, y_train)
#
#     # Make predictions on the test set
#     ry_pred = ridge_model1.predict(X_test)
#     mse = mean_squared_error(y_test, ry_pred)
#     rmse = np.sqrt(mse)
#
#     print(f'Alpha: {alpha}, Root Mean Squared Error (RMSE): {rmse}')


    ################################################

# plt.scatter(y_test, y_pred1)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.title('Actual vs. Predicted Values')
# # plt.show()


###############################################################

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create a Gradient Boosting Regressor
# gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#
# # Train the model
# gb_model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred_gb = gb_model.predict(X_test)
#
# # Evaluate the model performance
# mae_gb = mean_absolute_error(y_test, y_pred_gb)
# print(f'MAE Gradient Boosting: {mae_gb}')
# accuracy_percentage = (accurate_predictions.sum() / len(y_test)) * 100
# print(f"Accuracy percentage:{accuracy_percentage:.2f}")
# error_percentage = (np.mean(np.abs(y_test))/mae)
# print(f"Error Percentage: {error_percentage:.2f}")

##############################################################

# Example prediction for a new Pokémon
# new_pokemon_attributes = [[80, 70, 90, 70, 60]]  # Adjust these values as needed
# predicted_attack = model.predict(new_pokemon_attributes)
# print(f"Predicted Attack for the new Pokémon: {predicted_attack[0]:.2f}")