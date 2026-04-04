from groq import Groq
import os
import re


client = Groq(api_key=GROQ_API_KEY)

def generate_medical_advice(symptoms, diseases, age, gender, bmi, allergies, severe_issues):

    advice_dict = {}

    for disease, prob in diseases:

        prompt = f"""
        You are a careful medical assistant AI.

        Patient Profile:
        - Age: {age}
        - Gender: {gender}
        - BMI: {bmi}
        - Allergies: {allergies if allergies else "None"}
        - Severe Issues: {severe_issues if severe_issues else "None"}

        Symptoms:
        {symptoms}

        Predicted Disease:
        {disease} ({prob*100:.1f}% probability)

        Instructions:
        - Be practical and realistic (no overconfidence)
        - Only suggest safe, over-the-counter medicines
        - STRICTLY avoid medicines that conflict with allergies
        - If severe issues are present, increase risk level
        - If unsure, recommend doctor consultation

        Output format EXACTLY like this:

        Condition Summary:
        <short explanation>

        Risk Level:
        <Low / Medium / High with 1-line reason>

        Medicines:
        - <medicine 1>
        - <medicine 2>

        Precautions:
        - <point 1>
        - <point 2>

        Doctor Advice:
        <clear guidance: visit or not, and why>
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        advice = response.choices[0].message.content

        # clean formatting
        advice = re.sub(r"\*\*", "", advice)
        advice = re.sub(r"\*", "", advice)
        advice = re.sub(r"\n{3,}", "\n\n", advice)
        advice = advice.strip()

        advice_dict[disease] = advice

    return advice_dict
