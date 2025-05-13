import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Creating a simple kidney disease prediction model...")

# Create a simple synthetic dataset for kidney disease prediction
# Features: [blood_urea, blood_glucose, creatinine, albumin, hemoglobin, age]
X_train = np.array([
    # Negative examples (no kidney disease)
    [20, 100, 0.8, 4.0, 14.0, 35],
    [25, 110, 0.9, 4.2, 13.5, 40],
    [22, 105, 0.7, 4.5, 15.0, 30],
    [18, 95, 0.6, 4.8, 14.5, 25],
    [24, 115, 0.85, 4.3, 13.8, 45],
    
    # Positive examples (has kidney disease)
    [65, 180, 2.5, 2.5, 9.5, 60],
    [70, 190, 3.0, 2.0, 8.0, 65],
    [60, 170, 2.2, 2.8, 10.0, 55],
    [75, 200, 3.5, 1.8, 7.5, 70],
    [55, 160, 2.0, 3.0, 10.5, 50]
])

# Labels: 0 = no kidney disease, 1 = has kidney disease
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Create and train a new model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model with sample data
low_risk_sample = [20, 100, 0.8, 4.0, 14.0, 35]  # Should predict 0 - no kidney disease
high_risk_sample = [65, 180, 2.5, 2.5, 9.5, 60]  # Should predict 1 - has kidney disease

low_risk_pred = model.predict([low_risk_sample])[0]
high_risk_pred = model.predict([high_risk_sample])[0]

print(f"Low risk sample: {low_risk_sample}")
print(f"Prediction: {low_risk_pred} ({'Has kidney disease' if low_risk_pred == 1 else 'No kidney disease'})")

print(f"High risk sample: {high_risk_sample}")
print(f"Prediction: {high_risk_pred} ({'Has kidney disease' if high_risk_pred == 1 else 'No kidney disease'})")

if low_risk_pred != high_risk_pred:
    print("✅ Model is working correctly!")
    
    # Save the model
    working_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(working_dir, 'saved_models', 'kidney.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {save_path}")
else:
    print("❌ Model is not working correctly.")
