import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Step 1: Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)

    # Encode categorical variables
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    return df, label_encoders

# Step 2: Train the model
def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    return model

# Step 3: Make predictions
def predict_workout(model, new_user_input, X, label_encoders):
    input_data = []
    for key in new_user_input.keys():
        input_data.append(label_encoders[key].transform([new_user_input[key]])[0])

    predicted_plan_encoded = model.predict([input_data])[0]
    predicted_plan = X.index[predicted_plan_encoded]

    return predicted_plan

# Main workflow
if __name__ == "__main__":
    # Load and preprocess the data
    file_path = 'Functional_Fitness_Exercise_Database.xlsx'
    df, label_encoders = load_and_preprocess_data(file_path)
    
    # Train the model
    model = train_model(df)
    
    # Predict based on new user input
    new_user_input = {
        "focus_area": "chest",
        "equipment": "dumbbells",
        "fitness_level": "intermediate",
        "target_muscles": "pectorals"
    }

    suggested_plan = predict_workout(model, new_user_input, df, label_encoders)
    print(f"Suggested workout plan: {suggested_plan}")
