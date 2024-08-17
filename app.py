from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import json

# Load the machine learning models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
with open('countries.json', 'r') as file:
    countries = json.load(file)
with open('crops.json', 'r') as file:
    crops = json.load(file)

app = Flask(__name__)


@app.route('/')
def index():
    # Render the main page
    return render_template('index.html',countries=countries,crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    Year = request.form['Year']
    average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
    pesticides_tonnes = request.form['pesticides_tonnes']
    avg_temp = request.form['avg_temp']
    Area = request.form['Area']
    Item = request.form['Item']

    # Create a feature array
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    # Transform features using the preprocessor
    transformed_features = preprocessor.transform(features)
    # Predict using the decision tree regressor
    predicted_value = dtr.predict(transformed_features).reshape(1,-1)

    # Render the same page with the prediction result
    return render_template('index.html', crops=crops,countries=countries,predicted_value=predicted_value[0])

if __name__ == '__main__':
    app.run(debug=True)
