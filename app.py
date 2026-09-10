
from flask import Flask, render_template, request

import joblib
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the trained Medical Disease model
model = joblib.load(
    r'C:\Users\Rehan\OneDrive\Desktop\Data Science\6-Months-Data-Science\Medical-Disease\EDA\random_forest_model.pkl'
)

# Route for the main dashboard page
@app.route("/")
def index():
    return render_template('index.html')


# Route for the screening form page
@app.route("/medical_form.html")
def form():
    return render_template('medical_form.html')


# Route to handle form submission and make predictions
@app.route("/predict", methods=['POST'])
def predict():

    try:
        # Receive patient name
        patient_name = request.form.get('PatientName', 'Unknown')

        # Receive clinical features
        age = int(request.form['Age'])
        gender = int(request.form['Gender'])
        bmi = float(request.form['BMI'])
        bp = int(request.form['BloodPressure'])
        glucose = int(request.form['GlucoseLevel'])
        cholesterol = int(request.form['Cholesterol'])
        heart_rate = int(request.form['HeartRate'])
        smoking = int(request.form['Smoking'])
        alcohol = int(request.form['Alcohol'])
        physical_activity = int(request.form['PhysicalActivity'])
        family_history = int(request.form['FamilyHistory'])

        # Arrange features in the exact same order used during training
        features = np.array([[
            age,
            gender,
            bmi,
            bp,
            glucose,
            cholesterol,
            heart_rate,
            smoking,
            alcohol,
            physical_activity,
            family_history
        ]])

        # Predict directly using Random Forest
        prediction = model.predict(features)[0]

        # Get prediction probability
        probability = model.predict_proba(features)[0]

        # Disease mapping
        classes_dict = {
            1: "Healthy",
            2: "Pre-Diabetes",
            3: "Hypertension",
            4: "Heart Disease",
            5: "Diabetes"
        }

        # Convert prediction number into disease name
        condition = classes_dict.get(
            prediction,
            f"Condition Class: {prediction}"
        )

        # Calculate confidence
        predicted_index = list(model.classes_).index(prediction)
        confidence = probability[predicted_index] * 100

        # Final result
        result_text = (
            f"Patient: {patient_name} — "
            f"Result: {condition} — "
            f"Confidence: {confidence:.2f}%"
        )

        # Print result in VS Code terminal
        print(
            f"Data : Name : {patient_name}, "
            f"Age : {age}, "
            f"Result --> {condition}, "
            f"Confidence --> {confidence:.2f}%"
        )

        # Display result on webpage
        return render_template(
            'medical_form.html',
            prediction_text=result_text
        )

    except Exception as e:

        # Handle errors
        error_message = f"Error occurred: {str(e)}"

        print(f"\n[ERROR] {error_message}\n")

        return render_template(
            'medical_form.html',
            prediction_text=error_message
        )


# Run Flask development server
if __name__ == '__main__':
    app.run(debug=True)
