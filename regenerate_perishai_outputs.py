import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib

# 1. Define product types and storage conditions
def get_product_types_and_storage():
    product_types = ['apple', 'banana', 'milk', 'cheese', 'lettuce', 'chicken', 'yogurt']
    storage_conditions = ['cold', 'ambient', 'frozen']
    return product_types, storage_conditions

# 2. Simulate new delivery data with units in column names
def simulate_deliveries(product_types, storage_conditions, num_deliveries=20):
    sim_data = []
    for i in range(num_deliveries):
        product = random.choice(product_types)
        storage = random.choice(storage_conditions)
        time_in_transit = np.round(np.random.uniform(1, 48), 1)
        temperature = np.round(np.random.uniform(0, 25) if storage == 'cold' else np.random.uniform(15, 35), 1)
        humidity = np.round(np.random.uniform(40, 90), 1)
        sim_data.append({
            'delivery_id': f'DELV{i+1:03d}',
            'product_type': product,
            'time_in_transit (hours)': time_in_transit,
            'temperature_exposure (°C)': temperature,
            'humidity (%)': humidity,
            'storage_conditions': storage
        })
    return pd.DataFrame(sim_data)

# 3. Load encoders, scalers, and model from notebook pipeline
def load_pipeline():
    # Load training data to fit encoders/scalers
    train_df = pd.read_csv('perishai_dummy_data.csv')
    categorical_cols = ['product_type', 'storage_conditions']
    continuous_cols = ['time_in_transit (hours)', 'temperature_exposure (°C)', 'humidity (%)']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(train_df[categorical_cols])
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])
    model = joblib.load('shelf_life_model.pkl')
    return encoder, scaler, model

# 4. Preprocess and predict shelf life
def predict_shelf_life(sim_df, encoder, scaler, model):
    sim_X_encoded = encoder.transform(sim_df[['product_type', 'storage_conditions']])
    sim_X_scaled = scaler.transform(sim_df[['time_in_transit (hours)', 'temperature_exposure (°C)', 'humidity (%)']])
    sim_X_prepared = np.hstack([sim_X_encoded, sim_X_scaled])
    sim_df['predicted_shelf_life (days)'] = model.predict(sim_X_prepared)
    return sim_df

# 5. Save outputs
def save_outputs(sim_df):
    sim_df.to_csv('simulated_deliveries_with_predictions.csv', index=False)
    route_df_sorted = sim_df.sort_values(by='predicted_shelf_life (days)')
    route_df_sorted.reset_index(drop=True, inplace=True)
    route_df_sorted.to_csv('route_plan.csv', index=False)
    print('Files regenerated: simulated_deliveries_with_predictions.csv, route_plan.csv')

if __name__ == "__main__":
    product_types, storage_conditions = get_product_types_and_storage()
    sim_df = simulate_deliveries(product_types, storage_conditions)
    encoder, scaler, model = load_pipeline()
    sim_df = predict_shelf_life(sim_df, encoder, scaler, model)
    save_outputs(sim_df)
