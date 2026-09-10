# ============================================================
#                PLANTPULSE AI
#       Random Forest + Logistic Regression
#       3-Sensor ML Training
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_PATH = "PlantPulse_India_Crop_Stress_Dataset_75000.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. CREATE LIGHT PERCENTAGE
# ============================================================
#
# Dataset:
# Light_Intensity_Lux = 200 to approximately 1200 lux
#
# Arduino:
# lightLevel = 0 to 100 %
#
# We convert:
#
# 200 lux  -> 0%
# 1200 lux -> 100%
#
# This gives the ML model the same type of value
# that Arduino will provide.
# ============================================================

MIN_LUX = 200
MAX_LUX = 1200

df["Light_Percent"] = (
    (df["Light_Intensity_Lux"] - MIN_LUX)
    / (MAX_LUX - MIN_LUX)
) * 100

df["Light_Percent"] = df["Light_Percent"].clip(0, 100)


# ============================================================
# 4. FEATURES
# ============================================================
#
# These MUST match the data coming from Arduino.
#
# Arduino sends:
# DATA,soil,humidity,light,pump
#
# ML uses:
# Soil Moisture
# Humidity
# Light
#
# Pump is NOT an ML input.
# ============================================================

features = [
    "Soil_Moisture_Percent",
    "Humidity_Percent",
    "Light_Percent"
]

target = "Stress_Level"


X = df[features]
y = df[target]


# ============================================================
# 5. DISPLAY INFORMATION
# ============================================================

print("\n============================================")
print("FEATURES USED FOR MACHINE LEARNING")
print("============================================")

for feature in features:
    print("-", feature)


print("\nTarget:", target)

print("\nTarget classes:")
print(sorted(y.unique()))

print("\nClass distribution:")
print(y.value_counts().sort_index())


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(X.isnull().sum())

print("\nTarget missing values:")
print(y.isnull().sum())


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n============================================")
print("TRAIN / TEST SPLIT")
print("============================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 8. RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)


# ============================================================
# 9. RANDOM FOREST PREDICTION
# ============================================================

rf_predictions = rf_model.predict(X_test)


# ============================================================
# 10. RANDOM FOREST RESULTS
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)


print("\n============================================")
print("        RANDOM FOREST RESULTS")
print("============================================")

print(
    f"Accuracy: {rf_accuracy:.4f}"
)

print(
    f"Accuracy Percentage: {rf_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)


# ============================================================
# 11. LOGISTIC REGRESSION
# ============================================================
#
# StandardScaler is used because Logistic Regression
# works better when numerical features are scaled.
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model = Pipeline([
    
    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


logistic_model.fit(
    X_train,
    y_train
)


# ============================================================
# 12. LOGISTIC REGRESSION PREDICTION
# ============================================================

logistic_predictions = logistic_model.predict(
    X_test
)


# ============================================================
# 13. LOGISTIC REGRESSION RESULTS
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)


print("\n============================================")
print("      LOGISTIC REGRESSION RESULTS")
print("============================================")

print(
    f"Accuracy: {logistic_accuracy:.4f}"
)

print(
    f"Accuracy Percentage: {logistic_accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": rf_model.feature_importances_

})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n============================================")
print("      RANDOM FOREST FEATURE IMPORTANCE")
print("============================================")

print(importance.to_string(index=False))


# ============================================================
# 15. SAVE RANDOM FOREST
# ============================================================

joblib.dump(
    rf_model,
    "random_forest_model.pkl"
)


# ============================================================
# 16. SAVE LOGISTIC REGRESSION
# ============================================================

joblib.dump(
    logistic_model,
    "logistic_regression_model.pkl"
)


# ============================================================
# 17. SAVE FEATURE INFORMATION
# ============================================================

joblib.dump(
    features,
    "model_features.pkl"
)


# ============================================================
# 18. SAVE LIGHT CONVERSION INFORMATION
# ============================================================
#
# This is important later when we build the system.
# It records how the dataset's lux values were converted.
# ============================================================

light_conversion = {
    "min_lux": MIN_LUX,
    "max_lux": MAX_LUX,
    "output_min": 0,
    "output_max": 100
}

joblib.dump(
    light_conversion,
    "light_conversion.pkl"
)


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

print("\n============================================")
print("             MODEL COMPARISON")
print("============================================")

print(
    f"Random Forest        : {rf_accuracy * 100:.2f}%"
)

print(
    f"Logistic Regression  : {logistic_accuracy * 100:.2f}%"
)


if rf_accuracy > logistic_accuracy:

    print("\nBEST MODEL: Random Forest")

elif logistic_accuracy > rf_accuracy:

    print("\nBEST MODEL: Logistic Regression")

else:

    print("\nBoth models have the same accuracy.")


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("       MODELS SAVED SUCCESSFULLY")
print("============================================")

print("✓ random_forest_model.pkl")
print("✓ logistic_regression_model.pkl")
print("✓ model_features.pkl")
print("✓ light_conversion.pkl")

print("\nPlantPulse ML training completed!")
print("============================================")