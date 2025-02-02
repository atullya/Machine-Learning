import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('carprices.csv')

# One-hot encoding for categorical variable
dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df, dummies], axis='columns')
final = merged.drop(["Car Model", "Mercedez Benz C class"], axis='columns')

# Split data
X = final.drop('Sell Price($)', axis='columns')
y = final['Sell Price($)']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open('car_model.pkl', 'wb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    mileage = int(request.form['mileage'])
    age = int(request.form['age'])
    audi = int(request.form['audi'])
    bmw = int(request.form['bmw'])

    # Load the trained model
    model = pickle.load(open('car_model.pkl', 'rb'))

    # Make prediction
    prediction = model.predict([[mileage, age, audi, bmw]])
    return jsonify({'predicted_price': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
