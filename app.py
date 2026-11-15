from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("car_model.pkl")

fuel_encoder = joblib.load("fuel_encoder.pkl")
seller_encoder = joblib.load("seller_encoder.pkl")
trans_encoder = joblib.load("trans_encoder.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form['year'])
    present_price = float(request.form['present_price'])
    kms_driven = int(request.form['kms_driven'])

    fuel_type = request.form['fuel_type']
    seller_type = request.form['seller_type']
    transmission = request.form['transmission']

    owner = int(request.form['owner'])

    fuel_type = fuel_encoder.transform([fuel_type])[0]
    seller_type = seller_encoder.transform([seller_type])[0]
    transmission = trans_encoder.transform([transmission])[0]

    data = np.array([[
        year,
        present_price,
        kms_driven,
        fuel_type,
        seller_type,
        transmission,
        owner
    ]])

    prediction = model.predict(data)[0]

    return render_template(
        "result.html",
        prediction=round(prediction,2)
    )


if __name__ == "__main__":
    app.run(debug=True)