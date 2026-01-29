import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset (Download 'Crop_recommendation.csv' from Kaggle)
df = pd.read_csv('Crop_recommendation.csv')

# Features: N, P, K, temperature, humidity, ph, rainfall
X = df.drop('label', axis=1)
y = df['label']

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to a file
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as crop_model.pkl!")
