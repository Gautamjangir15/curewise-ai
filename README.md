# 🚀 CureWise AI

**AI-Powered Symptom-Based Health Advisory System**

CureWise AI is an intelligent healthcare assistant that analyzes user-reported symptoms to predict possible diseases and generate structured medical guidance using a hybrid approach of Machine Learning and Large Language Models (LLMs).

The system focuses on **controlled input, probabilistic predictions, and explainable advice**, offering a more reliable alternative to traditional symptom checkers.

---

## 🌐 Live Demo

> 🔗 https://curewise-ai.vercel.app/

---

## ✨ Key Features

### 🧠 Symptom-Based Disease Prediction  
Users select symptoms from a predefined list, ensuring clean and consistent input. The system predicts:

- **Top 3 possible diseases**
- **Probability score for each**

This probabilistic approach avoids overconfidence and reflects real-world diagnostic uncertainty.

---

### 🔒 Controlled Symptom Input — A Critical Design Choice  
By using a predefined symptom set instead of free text, the system:

- Eliminates noisy or invalid input  
- Improves model reliability  
- Aligns input format with training data  

This design decision is fundamental to maintaining prediction quality.

---

### 📊 Probabilistic Predictions  
The model outputs probabilities rather than a single label, providing a more realistic diagnostic perspective.

**Example output:**  
- Disease A → 62%  
- Disease B → 24%  
- Disease C → 14%

---

### 🤖 AI-Generated Medical Advice (LLM Layer)  
For each predicted disease, CureWise generates structured advice using an LLM (Groq API with LLaMA 3.1).  
The advice includes:

- Condition Summary  
- Risk Level  
- Suggested OTC Medicines  
- Precautions  
- Doctor Recommendation  

The system incorporates **allergy awareness**, **safety-first recommendations**, and **escalation logic** for severe symptoms.

---

### ⚖️ Context-Aware Risk Adjustment  
The system considers user-specific factors to tailor advice:

- Age  
- Gender  
- BMI  
- Severe symptoms  
- Allergies  

These factors influence the generated advice and risk assessment.

---

### 🧾 Explainable, Structured Output  
Every prediction is accompanied by clear, structured reasoning, making the system:

- Transparent  
- Interpretable  
- User-friendly  

---

### 🎨 Clean, Interactive UI  
- Symptom selection with a tag-based interface  
- Dark mode support  
- Expandable disease insights  
- Structured medical report view  

---

## 🖥 System Architecture

```
User Input (Controlled Symptoms)
        ↓
Feature Vectorization (TF-IDF)
        ↓
ML Model (Disease Prediction)
        ↓
Top-3 Probabilities
        ↓
LLM (Medical Advice Generation)
        ↓
Structured UI Output
```

---

## 🛠 Tech Stack

| Layer       | Technologies |
|-------------|--------------|
| **Backend** | Python, Flask |
| **Machine Learning** | Scikit-learn, TF-IDF Vectorizer, Classification Model |
| **Data Processing** | Pandas, NumPy |
| **AI Layer** | Groq API (LLaMA 3.1) |
| **Frontend** | HTML, CSS, JavaScript |

---

## 📂 Project Structure

```
curewise/
│
├── app.py                      # Flask application
├── advisor.py                  # LLM advice generation logic
├── disease_model.pkl           # Trained ML model
├── vectorizer.pkl              # TF-IDF vectorizer
├── label_encoder.pkl           # Encoded disease labels
├── curewise_dataset_shuffled.csv
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
```

---

## 📋 Example Workflow

1. User selects symptoms (e.g., *fever, cough*)  
2. System converts input into feature vector  
3. ML model predicts top 3 diseases with probabilities  
4. LLM generates structured medical advice  
5. User receives a complete health report  

---

## 🎯 Why This Project Matters

Most symptom checkers suffer from:

- Noisy user input  
- Overconfident predictions  
- Lack of explainability  

CureWise AI addresses these issues by combining:

- **Controlled input system**  
- **Probabilistic ML predictions**  
- **LLM-based explainable advice**  
- **Context-aware risk analysis**  

The result is a more **reliable, transparent, and user-friendly** healthcare assistant.

---

## ⚠️ Limitations

- Uses shallow text representation (TF-IDF)  
- Not a substitute for professional medical diagnosis  
- Prediction quality depends on dataset completeness and accuracy  

---

## 🔮 Future Improvements

- Replace TF-IDF with semantic embeddings (e.g., Sentence Transformers)  
- Add fuzzy matching for symptom input  
- Integrate a medical knowledge base  
- Implement confidence threshold filtering  
- Add patient history tracking  
- Deploy as a real-time API  

---

## 👨‍💻 Author

**Gautam Jangir**  
