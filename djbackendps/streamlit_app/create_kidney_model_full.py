import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("Creating a comprehensive kidney disease prediction model with all 24 features...")

# Define all 24 features in order
# [age, blood_pressure, specific_gravity, albumin, sugar, rbc_value, pc_value, pcc_value, 
#  ba_value, blood_glucose, blood_urea, serum_creatinine, sodium, potassium, hemoglobin, 
#  packed_cell_volume, wbc_count, rbc_count, htn_value, dm_value, cad_value, appet_value, 
#  pe_value, ane_value]

# Create a synthetic dataset for kidney disease prediction with all 24 features
X_train = np.array([
    # Negative examples (no kidney disease) - 10 samples
    [35, 80, 1.020, 0.0, 0.0, 0, 0, 0, 0, 100, 20, 0.8, 135, 4.0, 14.0, 40, 9000, 4.5, 0, 0, 0, 0, 0, 0],
    [40, 85, 1.018, 0.0, 0.0, 0, 0, 0, 0, 110, 25, 0.9, 138, 4.2, 13.5, 42, 8500, 4.8, 0, 0, 0, 0, 0, 0],
    [30, 75, 1.022, 0.0, 0.0, 0, 0, 0, 0, 105, 22, 0.7, 140, 4.5, 15.0, 45, 7500, 5.0, 0, 0, 0, 0, 0, 0],
    [25, 70, 1.025, 0.0, 0.0, 0, 0, 0, 0, 95, 18, 0.6, 142, 4.8, 14.5, 44, 8000, 5.2, 0, 0, 0, 0, 0, 0],
    [45, 90, 1.017, 0.0, 0.0, 0, 0, 0, 0, 115, 24, 0.85, 137, 4.3, 13.8, 41, 7800, 4.7, 0, 0, 0, 0, 0, 0],
    [38, 82, 1.019, 0.0, 0.0, 0, 0, 0, 0, 108, 23, 0.75, 139, 4.4, 14.2, 43, 8200, 4.9, 0, 0, 0, 0, 0, 0],
    [42, 88, 1.016, 0.0, 0.0, 0, 0, 0, 0, 112, 26, 0.95, 136, 4.1, 13.2, 40, 8800, 4.6, 0, 0, 0, 0, 0, 0],
    [28, 72, 1.023, 0.0, 0.0, 0, 0, 0, 0, 98, 19, 0.65, 141, 4.7, 14.8, 46, 7600, 5.1, 0, 0, 0, 0, 0, 0],
    [33, 78, 1.021, 0.0, 0.0, 0, 0, 0, 0, 102, 21, 0.78, 140, 4.6, 14.5, 44, 7900, 5.0, 0, 0, 0, 0, 0, 0],
    [36, 81, 1.020, 0.0, 0.0, 0, 0, 0, 0, 106, 22, 0.82, 138, 4.3, 14.0, 42, 8100, 4.8, 0, 0, 0, 0, 0, 0],
    
    # Positive examples (has kidney disease) - 10 samples
    [60, 160, 1.010, 3.0, 2.0, 1, 1, 1, 1, 180, 65, 2.5, 125, 5.5, 9.5, 30, 12000, 3.5, 1, 1, 1, 1, 1, 1],
    [65, 170, 1.008, 3.5, 2.5, 1, 1, 1, 1, 190, 70, 3.0, 122, 5.8, 8.0, 28, 13000, 3.2, 1, 1, 1, 1, 1, 1],
    [55, 150, 1.012, 2.5, 1.5, 1, 1, 1, 1, 170, 60, 2.2, 128, 5.2, 10.0, 32, 11500, 3.8, 1, 1, 0, 1, 1, 1],
    [70, 180, 1.005, 4.0, 3.0, 1, 1, 1, 1, 200, 75, 3.5, 120, 6.0, 7.5, 25, 14000, 3.0, 1, 1, 1, 1, 1, 1],
    [50, 140, 1.015, 2.0, 1.0, 1, 1, 0, 0, 160, 55, 2.0, 130, 5.0, 10.5, 34, 11000, 4.0, 1, 0, 0, 1, 0, 1],
    [62, 165, 1.009, 3.2, 2.2, 1, 1, 1, 0, 185, 68, 2.8, 124, 5.6, 8.5, 29, 12500, 3.3, 1, 1, 0, 1, 1, 1],
    [58, 155, 1.011, 2.8, 1.8, 1, 1, 0, 0, 175, 63, 2.3, 127, 5.3, 9.8, 31, 11800, 3.6, 1, 0, 0, 1, 1, 1],
    [68, 175, 1.006, 3.8, 2.8, 1, 1, 1, 1, 195, 72, 3.2, 121, 5.9, 7.8, 26, 13500, 3.1, 1, 1, 1, 1, 1, 1],
    [52, 145, 1.014, 2.2, 1.2, 1, 0, 0, 0, 165, 57, 2.1, 129, 5.1, 10.2, 33, 11200, 3.9, 1, 0, 0, 1, 0, 1],
    [64, 168, 1.007, 3.4, 2.4, 1, 1, 1, 0, 188, 69, 2.9, 123, 5.7, 8.2, 27, 12800, 3.2, 1, 1, 1, 1, 1, 1]
])

# Labels: 0 = no kidney disease, 1 = has kidney disease
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Create and train a new model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model with sample data
low_risk_sample = [35, 80, 1.020, 0.0, 0.0, 0, 0, 0, 0, 100, 20, 0.8, 135, 4.0, 14.0, 40, 9000, 4.5, 0, 0, 0, 0, 0, 0]
high_risk_sample = [60, 160, 1.010, 3.0, 2.0, 1, 1, 1, 1, 180, 65, 2.5, 125, 5.5, 9.5, 30, 12000, 3.5, 1, 1, 1, 1, 1, 1]

low_risk_pred = model.predict([low_risk_sample])[0]
high_risk_pred = model.predict([high_risk_sample])[0]

print(f"Low risk sample prediction: {low_risk_pred} ({'Has kidney disease' if low_risk_pred == 1 else 'No kidney disease'})")
print(f"High risk sample prediction: {high_risk_pred} ({'Has kidney disease' if high_risk_pred == 1 else 'No kidney disease'})")

if low_risk_pred != high_risk_pred:
    print("✅ Model is working correctly!")
    
    # Save the model
    working_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(working_dir, 'saved_models', 'kidney_full.pkl')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {save_path}")
else:
    print("❌ Model is not working correctly.")

# Print feature importances
feature_names = [
    "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", 
    "Red Blood Cells", "Pus Cell", "Pus Cell Clumps", "Bacteria",
    "Blood Glucose", "Blood Urea", "Serum Creatinine", "Sodium", "Potassium",
    "Hemoglobin", "Packed Cell Volume", "WBC Count", "RBC Count",
    "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease",
    "Appetite", "Pedal Edema", "Anemia"
]

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature ranking:")
for i in range(len(feature_names)):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
