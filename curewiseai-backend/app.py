from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
from advisor import generate_medical_advice
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "../curewiseai-frontend/templates"),
    static_folder=os.path.join(BASE_DIR, "../curewiseai-frontend/static")
)
CORS(app)

# load ML components
model = pickle.load(open("new_disease_model.pkl","rb"))
vectorizer = pickle.load(open("new_vectorizer.pkl","rb"))
le = pickle.load(open("new_label_encoder.pkl","rb"))

df = pd.read_csv('curewise_dataset_shuffled.csv')
SYMPTOM_SET = set()

for s in df["symptoms"].dropna():
    
    if not isinstance(s, str):   # 🔒 important fix
        continue

    parts = [p.strip().lower() for p in s.split(",")]

    for p in parts:
        if p:
            SYMPTOM_SET.add(p)

def is_valid_symptom_input(user_input):

    user_input = user_input.lower()

    # normalize input
    user_input = user_input.replace(" and ", ",")
    user_input = user_input.replace(".", ",")

    parts = [p.strip() for p in user_input.split(",") if p.strip()]

    if len(parts) == 0:
        return False

    valid_matches = 0

    for user_symptom in parts:
        for real_symptom in SYMPTOM_SET:
            if user_symptom in real_symptom or real_symptom in user_symptom:
                valid_matches += 1
                break

    # at least 1 real symptom must match
    return valid_matches >= 1            

def predict_top3(symptoms):

    vec = vectorizer.transform([symptoms])

    probs = model.predict_proba(vec)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]

    diseases = le.inverse_transform(top3_idx)

    results = []

    for d,i in zip(diseases, top3_idx):
        results.append((d, float(probs[i])))

    return results


@app.route("/")
def home():
    return render_template("index.html", symptoms_list=sorted(SYMPTOM_SET))


@app.route("/result", methods=["POST"])
def result():

    symptoms = request.form["symptoms"].strip().lower()
    
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    age = request.form["age"]
    gender = request.form["gender"]
    name = request.form["name"]

    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)

    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_status = "Healthy"
        bmi_color = "green"
    else:
        bmi_status = "Overweight"
        bmi_color = "red"
    

    predictions = predict_top3(symptoms)

    advice = generate_medical_advice(symptoms,predictions,age,gender,bmi,request.form["allergies"],request.form["severe_issues"])
    
    best_disease = predictions[0][0]

    return render_template(
    "result.html",
    predictions=predictions,
    advice=advice,
    best_disease=best_disease,
    bmi=bmi,
    bmi_status=bmi_status,
    bmi_color=bmi_color,
    name=name,
    age=age,
    gender=gender

)
@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.json

    symptoms = data["symptoms"].strip().lower()
    height = float(data["height"])
    weight = float(data["weight"])
    age = data["age"]
    gender = data["gender"]

    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)
    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_status = "Healthy"
        bmi_color = "green"
    else:
        bmi_status = "Overweight"
        bmi_color = "red"

    predictions = predict_top3(symptoms)

    advice = generate_medical_advice(
        symptoms,
        predictions,
        age,
        gender,
        bmi,
        data.get("allergies"),
        data.get("severe_issues")
    )

    best_disease = predictions[0][0]

    return jsonify({
        "predictions": predictions,
        "advice": advice,
        "best_disease": best_disease,
        "bmi": bmi,
        "bmi_status":bmi_status,
        "bmi_color":bmi_color,
        "age": age,
        "gender": gender,
        "name": data.get("name")
    })

if __name__ == "__main__":
    app.run(debug=True)
