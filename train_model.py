import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv("car_data.csv")

# Encode categorical columns
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

df['Fuel_Type'] = fuel_encoder.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = seller_encoder.fit_transform(df['Seller_Type'])
df['Transmission'] = trans_encoder.fit_transform(df['Transmission'])

# Features and Target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

score = r2_score(y_test, pred)

print("Model Trained Successfully")
print("R² Score:", round(score,4))

# Save model and encoders
joblib.dump(model,"car_model.pkl")
joblib.dump(fuel_encoder,"fuel_encoder.pkl")
joblib.dump(seller_encoder,"seller_encoder.pkl")
joblib.dump(trans_encoder,"trans_encoder.pkl")

print("Files Saved Successfully")