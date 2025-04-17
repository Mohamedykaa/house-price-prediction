# 🔍 Project: House Price Prediction using Random Forest

"""
Team Members:
1. Mohamed Yasser Karam Abd Elshafy - C2201922
2. Abdelrahman Mostafa Mohamed - C2201767
3. Mohamed Ayman Mohamed Khalif - C2200071
4. Mohamed Ahmed Mohamed Tawfik - C2200455
5. Ahmed Mohamed Shadid Haddad - C2200882
"""

# ✅ 1. Load the data
from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame

# ✅ 2. Explore the data
print(df.head())
print(df.info())
print(df.describe())

# ✅ 3. Basic visual analysis
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']])
plt.show()

# ✅ 4. Data preprocessing
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df.drop('MedHouseVal', axis=1))
y = df['MedHouseVal']

# ✅ 5. Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 6. Try different models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVR": SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)

# ✅ 7. Evaluate performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"{name} Evaluation")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2:", r2_score(y_test, y_pred))
    print("-" * 30)

for name, model in models.items():
    evaluate_model(name, model)

# ✅ 8. Improve Random Forest performance
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(), params, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

evaluate_model("Tuned Random Forest", best_model)

# ✅ 9. Save the model
import joblib

joblib.dump(best_model, 'random_forest_model.pkl')

# ✅ 10. Predict new values
sample_input = X_test[0].reshape(1, -1)
prediction = best_model.predict(sample_input)
print("Sample Prediction:", prediction)

# ✅ 11. Plot the results
import numpy as np

y_pred = best_model.predict(X_test)
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.grid(True)
plt.show()

# ✅ 12. Cross-Validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print("Cross-Validated R² Scores:", cv_scores)
print("Average R² Score:", cv_scores.mean())

# ✅ 13. Feature Importance Visualization
importances = best_model.feature_importances_
features = df.drop('MedHouseVal', axis=1).columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

# ✅ 14. Save performance report
with open("model_report.txt", "w") as f:
    f.write(f"Best model: Random Forest\n")
    f.write(f"MSE: {mean_squared_error(y_test, y_pred)}\n")
    f.write(f"MAE: {mean_absolute_error(y_test, y_pred)}\n")
    f.write(f"R^2: {r2_score(y_test, y_pred)}\n")

# ✅ 15. Create Web API using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']  # Expects a list of features
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
