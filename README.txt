Project artifacts produced in /mnt/data/periop_project

Files:
- perioperative_updated_1200.csv     : Synthetic dataset with Recommended_Medication, Dosage, Duration
- complication_predictor_pipeline.pkl: Trained sklearn pipeline (preprocessing + RandomForest)
- medication_map.csv                 : Example mapping of complication -> medication/dosage/duration
- backend_app.py                     : Simple Flask backend template that loads the model
- frontend_index.html                : Simple static frontend that posts to /api/predict (assumes proxy)

How to run (locally):
1. Create a virtualenv and install requirements:
   pip install flask scikit-learn joblib pandas

2. Start backend (this serves prediction endpoint):
   python backend_app.py
   The app will run on port 5000 and expose /predict

Note: In the frontend code, it posts to /api/predict; you may need to setup a proxy or change it to the full URL http://localhost:5000/predict

Model evaluation:
- Test accuracy: 0.3375

