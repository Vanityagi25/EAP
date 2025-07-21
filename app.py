# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Mapping for categorical values
mappings = {
    "BusinessTravel": {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2
    },
    "Department": {
        "Sales": 0,
        "Research & Development": 1,
        "Human Resources": 2
    },
    "EducationField": {
        "Life Sciences": 0,
        "Medical": 1,
        "Marketing": 2,
        "Technical Degree": 3,
        "Human Resources": 4,
        "Other": 5
    },
    "Gender": {
        "Male": 0,
        "Female": 1
    },
    "JobRole": {
        "Sales Executive": 0,
        "Research Scientist": 1,
        "Laboratory Technician": 2,
        "Manufacturing Director": 3,
        "Healthcare Representative": 4,
        "Manager": 5,
        "Sales Representative": 6,
        "Research Director": 7,
        "Human Resources": 8
    },
    "MaritalStatus": {
        "Single": 0,
        "Married": 1,
        "Divorced": 2
    },
    "OverTime": {
        "Yes": 1,
        "No": 0
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from form
        data = request.form

        # Map values
        BusinessTravel = mappings["BusinessTravel"][data["BusinessTravel"]]
        Department = mappings["Department"][data["Department"]]
        EducationField = mappings["EducationField"][data["EducationField"]]
        Gender = mappings["Gender"][data["Gender"]]
        JobRole = mappings["JobRole"][data["JobRole"]]
        MaritalStatus = mappings["MaritalStatus"][data["MaritalStatus"]]
        MonthlyIncome = int(data["MonthlyIncome"])
        OverTime = mappings["OverTime"][data["OverTime"]]
        TotalWorkingYears = int(data["TotalWorkingYears"])

        # Final input for model
        final_features = [[
            BusinessTravel, Department, EducationField, Gender,
            JobRole, MaritalStatus, MonthlyIncome, OverTime, TotalWorkingYears
        ]]

        prediction = model.predict(final_features)
        output = 'No' if prediction[0] == 1 else 'Yes'

        return render_template('index.html', prediction_text='Will employee attrite? → {}'.format(output))

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
