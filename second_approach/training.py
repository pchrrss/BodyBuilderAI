import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('data/fitness_plan_4m.csv') 

# Prepare the features and target variable
X = df.drop(columns=['fitness_plan'])
y = df['fitness_plan']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=['age_range', 'body_type', 'goal', 'body_fat_range', 'focus_area', 'equipment'])

# Encode the target variable (fitness plans) using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Decode the predicted labels back to original fitness plans
y_pred_text = label_encoder.inverse_transform(y_pred)
y_test_text = label_encoder.inverse_transform(y_test)

# Evaluate the model
print(classification_report(y_test_text, y_pred_text))

# Save the model and label encoder
import pickle
with open('model/fitness_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('model/label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

X_columns = X.columns.tolist()
with open('model/X_columns.pkl', 'wb') as columns_file:
    pickle.dump(X_columns, columns_file)

print("Model and label encoder saved successfully.")
