import pandas as pd
import pickle

# Step 1: Load the Saved Model, Label Encoder, and Feature Names
with open('model/fitness_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('model/label_encoder.pkl', 'rb') as encoder_file:
    loaded_label_encoder = pickle.load(encoder_file)

with open('model/X_columns.pkl', 'rb') as columns_file:
    X_columns = pickle.load(columns_file)

# Step 2: Prepare New Data for Prediction
new_data = pd.DataFrame({
    'age_range': ['30-39'],
    'body_type': ['average'],
    'goal': ['gain muscle mass'],
    'body_fat_range': ['20-24%'],
    'focus_area': ['chest'],
    'fitness_level': [7],
    'equipment': ['basic equipment'],
    'times_per_week': [4]
})

# Convert categorical variables into dummy/indicator variables
new_data_processed = pd.get_dummies(new_data, columns=['age_range', 'body_type', 'goal', 'body_fat_range', 'focus_area', 'equipment'])

# Ensure the new data has the same columns as the training data
for col in X_columns:
    if col not in new_data_processed:
        new_data_processed[col] = 0

new_data_processed = new_data_processed[X_columns]  # Reorder columns to match training data

# Step 3: Make Predictions
predictions_encoded = loaded_model.predict(new_data_processed)

# Convert the numerical prediction back to the original fitness plan text using the LabelEncoder
predictions = loaded_label_encoder.inverse_transform(predictions_encoded)

# Step 4: Output the Prediction
print("Predicted Fitness Plan:", predictions[0])
