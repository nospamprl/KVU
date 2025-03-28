# Import necessary libraries
import pandas as pd
import joblib
import numpy as np
import os
from flask import Flask, request, jsonify
import xgboost as xgb
from flask_cors import CORS
from datetime import datetime


app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": ["https://advantcapital.com.mx"]}})

# Define absolute paths for files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best_xgboost_model.pkl')
mapping_file = os.path.join(BASE_DIR, 'mappings.xlsx')

# Load the trained model
model = joblib.load(model_path)

# Extract original feature names from the model
original_feature_names = model.get_booster().feature_names

# Load mappings from new sheet names
make_mapping = pd.read_excel(mapping_file, sheet_name='Marca').set_index('Marca')['Índice'].to_dict()
model_mapping = pd.read_excel(mapping_file, sheet_name='Modelo').set_index('Modelo')['Índice'].to_dict()
version_mapping = pd.read_excel(mapping_file, sheet_name='Version').set_index('Version')['Índice'].to_dict()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract and convert text inputs to indices based on mappings
    make = make_mapping.get(data.get('Make'), -1)
    car_model = model_mapping.get(data.get('Model'), -1)
    version = version_mapping.get(data.get('Version'), -1)

    if -1 in [make, car_model, version]:
        return jsonify({'error': 'Invalid selection'}), 400

    # Prepare data for prediction
    #age = 2025 - int(data['Year']) #  **********************************************************  Cambiar cada año calendario
    age = datetime.now().year - int(data['Year'])
    mileage = float(data['Mileage'])
    list_price = float(data['ListPrice'])

    input_data = pd.DataFrame([[age, mileage, make, car_model, version, list_price]],
                              columns=original_feature_names).astype(float)

    # Make prediction
    predicted_price = model.predict(input_data)[0]
    return jsonify({'predicted_price': float(round(predicted_price, 2))})

@app.route('/get-options', methods=['GET'])
def get_options():
    try:
        # Load the Catalogo sheet
        catalog_df = pd.read_excel(mapping_file, sheet_name='Catalogo')

        # Extract makes, models, and versions
        makes = catalog_df['Marca'].dropna().unique().tolist()

        # Build model and version relationships
        model_map = catalog_df[['Marca', 'Modelo']].drop_duplicates().to_dict('records')
        version_map = catalog_df[['Modelo', 'Version']].drop_duplicates().to_dict('records')

        return jsonify({
            'makes': makes,
            'models': model_map,
            'versions': version_map
        })
    except Exception as e:
        print("Error fetching options:", str(e))
        return jsonify({'error': 'Failed to load options'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
