import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("perioperative_updated_1200.csv")

# Target
df = df.dropna(subset=['Complication'])
y = df['Complication']

# Features (drop ID and non-predictive fields)
X = df.drop(columns=['Patient_ID', 'Complication', 'Risk_Score', 
                     'Recommended_Medication', 'Dosage', 'Duration'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

# Categorical and numerical features
categorical = ['Gender', 'Diabetes', 'Hypertension', 'Heart_Disease', 
               'Surgery_Type', 'Vital_Instability']
numerical = [col for col in X.columns if col not in categorical]

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, "complication_predictor_pipeline.pkl")

print("✅ Model trained and saved as complication_predictor_pipeline.pkl")
