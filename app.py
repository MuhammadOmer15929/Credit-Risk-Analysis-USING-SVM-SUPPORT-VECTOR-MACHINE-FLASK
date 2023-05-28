from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("svm_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the input values from the form
        person_age = int(request.form["person_age"])
        person_income = float(request.form["person_income"])
        person_home_ownership = request.form["person_home_ownership"]
        person_emp_length = int(request.form["person_emp_length"])
        loan_intent = request.form["loan_intent"]
        loan_grade = request.form["loan_grade"]
        loan_amnt = float(request.form["loan_amnt"])
        loan_int_rate = float(request.form["loan_int_rate"])
        cb_person_default_on_file = request.form["cb_person_default_on_file"]
        cb_person_cred_hist_length = int(request.form["cb_person_cred_hist_length"])

        # Preprocess the input data
        data = pd.DataFrame({
            "person_age": [person_age],
            "person_income": [person_income],
            "person_home_ownership": [person_home_ownership],
            "person_emp_length": [person_emp_length],
            "loan_intent": [loan_intent],
            "loan_grade": [loan_grade],
            "loan_amnt": [loan_amnt],
            "loan_int_rate": [loan_int_rate],
            "cb_person_default_on_file": [cb_person_default_on_file],
            "cb_person_cred_hist_length": [cb_person_cred_hist_length]
        })

        for column, le in label_encoders.items():
            data[column] = le.transform(data[column])

        # Perform prediction using the loaded model
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]  # Probability of positive class

        # Convert prediction to risk label
        risk_label = "Yes" if prediction[0] == 1 else "No"

        # Return the prediction to the user
        return render_template("index.html", prediction=risk_label, probability=probability)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
