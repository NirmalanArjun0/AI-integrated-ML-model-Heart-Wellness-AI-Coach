import json
import tempfile
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Try importing Ollama (LLM)
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False

# mention Paths 
ROOT = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT / "assets" / "style.css"
CSS = CSS_PATH.read_text(encoding="utf-8")

MODEL_PATH = ROOT / "models" / "heartcoach_rf.pkl"
ENC_PATH = ROOT / "models" / "label_encoder.pkl"
META_PATH = ROOT / "docs" / "used_features.json"
TIPS_PATH = ROOT / "rag" / "medical_guidelines.txt"

# 
# Load model + metadata 
if not MODEL_PATH.exists() or not META_PATH.exists() or not ENC_PATH.exists():
    raise RuntimeError(
        "Model or metadata not found. Run 'python notebooks/01_train_model.py' first from the project root."
    )

pipe = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENC_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
num_features = meta["numeric_features"]
cat_features = meta["categorical_features"]
classes = meta["classes"]


# Simple RAG over guidelines 
# Reads the guidelines text file.
#Splits it into paragraphs
#Builds a TF-IDF vector for each paragraph → numeric representation for text.

class SimpleRAG:
    def __init__(self, path: Path):
        txt = path.read_text(encoding="utf-8")
        self.docs = [b.strip() for b in txt.split("\n\n") if b.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vecs = self.vectorizer.fit_transform(self.docs)

    def query(self, q: str, k: int = 3):
        qv = self.vectorizer.transform([q])
        sims = linear_kernel(qv, self.doc_vecs).flatten()
        idx = sims.argsort()[::-1][:k]
        return [self.docs[i] for i in idx]


rag = SimpleRAG(TIPS_PATH)

# Small ontology (risk concepts)
#This is a small dictionary of knowledge.
#For each risk type (High BP, Smoking etc.) ihave stored a short explanation sentence.

HEALTH_ONTOLOGY = {
    "Heart_Disease_Risk": {
        "High_Blood_Pressure": "Systolic and diastolic blood pressure above normal increase heart strain.",
        "High_BMI": "Excess body weight increases risk of hypertension, diabetes, and heart disease.",
        "Smoking": "Smoking damages blood vessels and significantly increases heart disease risk.",
        "Low_Exercise": "Low physical activity is linked to poorer cardiovascular fitness.",
        "Diabetes": "Diabetes is a major risk factor for heart disease.",
        "High_Alcohol": "High alcohol intake can raise blood pressure and damage the heart.",
        "Poor_Sleep": "Short sleep and poor-quality sleep can affect blood pressure and metabolism.",
    }
}


#Rule-based Agent suggestions
# Then it reads values from the row:
def agent_suggestions(row: dict):
    recs = []
    explanatory = []

    sleep = float(row.get("Hours_of_Sleep", 0) or 0)
    steps = float(row.get("Daily_Steps", 0) or 0)
    hr = float(row.get("Heart_Rate", 0) or 0)
    ex_hrs = float(row.get("Exercise_Hours_per_Week", 0) or 0)
    bmi = float(row.get("BMI", 0) or 0)
    sys = float(row.get("Systolic", 0) or 0)
    dia = float(row.get("Diastolic", 0) or 0)
    gender = str(row.get("Gender", "")).lower()
    alcohol_wk = float(row.get("Alcohol_Consumption_per_Week", 0) or 0)
    smoker = str(row.get("Smoker", "")).lower()
    diabetic = str(row.get("Diabetic", "")).lower()

    # Sleep
    if sleep < 7:
        recs.append(
            "Try to increase your sleep towards 7–9 hours per night with a regular routine."
        )
        explanatory.append("Poor_Sleep")

    # Exercise
    if ex_hrs < 2.5:
        recs.append(
            "Increase weekly exercise towards at least 150 minutes of moderate activity (for example brisk walking)."
        )
        explanatory.append("Low_Exercise")

    # Steps
    if steps < 8000:
        recs.append("Add 1,000–2,000 extra steps per day using short walks or standing breaks.")
        explanatory.append("Low_Exercise")

    # Resting heart rate
    if hr > 80:
        recs.append(
            "Your resting heart rate is a bit high. Improve sleep, stress management, and light cardio."
        )

    # BMI
    if bmi >= 30:
        recs.append(
            "Your BMI suggests obesity. Small daily calorie reductions and regular activity can help long-term weight loss."
        )
        explanatory.append("High_BMI")
    elif 25 <= bmi < 30:
        recs.append(
            "Your BMI is in the overweight range. A small calorie deficit and more activity could be helpful."
        )
        explanatory.append("High_BMI")

    # Blood pressure
    if sys and dia:
        if sys >= 180 or dia >= 120:
            recs.append(
                "Your blood pressure is in a very high range. Seek urgent medical care if you feel unwell."
            )
            explanatory.append("High_Blood_Pressure")
        elif sys >= 140 or dia >= 90:
            recs.append(
                "Your blood pressure is in the hypertension range. Please discuss with a clinician and focus on lifestyle changes."
            )
            explanatory.append("High_Blood_Pressure")
        elif (130 <= sys <= 139) or (80 <= dia <= 89):
            recs.append(
                "Your blood pressure is in the elevated range. Add exercise, reduce salt, manage stress, and monitor readings."
            )
            explanatory.append("High_Blood_Pressure")

    # Alcohol
    if gender.startswith("male") and alcohol_wk > 14:
        recs.append(
            "Your weekly alcohol intake is above common health guidance for men. Try to gradually reduce your intake."
        )
        explanatory.append("High_Alcohol")
    if gender.startswith("female") and alcohol_wk > 7:
        recs.append(
            "Your weekly alcohol intake is above common health guidance for women. Reducing intake can improve heart health."
        )
        explanatory.append("High_Alcohol")

    # Smoking
    if smoker.startswith("yes"):
        recs.append(
            "Stopping smoking is one of the strongest ways to lower heart disease risk. Consider a structured quit plan."
        )
        explanatory.append("Smoking")

    # Diabetes
    if diabetic.startswith("yes"):
        recs.append(
            "Because of diabetes, controlling blood sugar, blood pressure, and cholesterol is very important."
        )
        explanatory.append("Diabetes")

    if not recs:
        recs.append(
            "Overall, your profile looks reasonable. Maintain healthy habits and regular check-ups."
        )

    # Ontology notes
    onto_notes = []
    for key in set(explanatory):
        desc = HEALTH_ONTOLOGY["Heart_Disease_Risk"].get(key)
        if desc:
            onto_notes.append(f"- {key.replace('_', ' ')}: {desc}")

    return recs, onto_notes


#  Simple rule-based lifestyle risk level 
def rule_risk_level(row: dict) -> str:
    """
    Simple scoring to classify lifestyle risk as LOW / MODERATE / HIGH
    based on BMI, blood pressure, smoking, diabetes, sleep, steps, exercise, alcohol.
    """
    score = 0

    bmi = float(row.get("BMI", 0) or 0)
    sys = float(row.get("Systolic", 0) or 0)
    dia = float(row.get("Diastolic", 0) or 0)
    sleep = float(row.get("Hours_of_Sleep", 0) or 0)
    steps = float(row.get("Daily_Steps", 0) or 0)
    ex_hrs = float(row.get("Exercise_Hours_per_Week", 0) or 0)
    alcohol = float(row.get("Alcohol_Consumption_per_Week", 0) or 0)
    gender = str(row.get("Gender", "")).lower()
    smoker = str(row.get("Smoker", "")).lower()
    diabetic = str(row.get("Diabetic", "")).lower()

    # BMI
    if bmi >= 30:
        score += 3
    elif 25 <= bmi < 30:
        score += 1

    # Blood pressure
    if sys >= 160 or dia >= 100:
        score += 3
    elif sys >= 140 or dia >= 90:
        score += 2
    elif sys >= 130 or dia >= 85:
        score += 1

    # Sleep
    if sleep < 5:
        score += 2
    elif sleep < 7:
        score += 1

    # Exercise
    if ex_hrs < 1:
        score += 2
    elif ex_hrs < 2.5:
        score += 1

    # Steps
    if steps < 5000:
        score += 2
    elif steps < 8000:
        score += 1

    # Alcohol
    if gender.startswith("male") and alcohol > 14:
        score += 2
    if gender.startswith("female") and alcohol > 7:
        score += 2

    # Smoking
    if smoker.startswith("yes"):
        score += 3

    # Diabetes
    if diabetic.startswith("yes"):
        score += 3

    # Map score to level
    if score >= 8:
        return "HIGH"
    elif score >= 4:
        return "MODERATE"
    else:
        return "LOW"


# ---------- Ollama helper (AI Coach – main explanation) ----------
def call_ollama_coach(row: dict, pred_text: str, tips_text: str) -> str:
    """
    Ask the local Ollama model to explain the situation in simple language.
    """
    if not HAS_OLLAMA:
        return (
            "LLM explanation is not available because the 'ollama' Python package "
            "is not installed in this environment."
        )

    system_prompt = (
        "You are a careful, calm medical AI coach. "
        "You explain heart disease risk and lifestyle changes in very simple English. "
        "You must NOT give strict diagnosis, only educational guidance. "
        "Always remind the user to consult a doctor for medical decisions."
    )

    summary_lines = [f"{k}: {v}" for k, v in row.items()]

    user_prompt = f"""
Here is the structured health profile of a person:

{chr(10).join(summary_lines)}

The ML model prediction and probabilities are:

{pred_text}

The rule-based suggestions and RAG tips are:

{tips_text}

Please:
1. Briefly explain what this risk level means in simple English.
2. Point out 3–5 main things they should focus on (sleep, BP, BMI, steps, smoking, alcohol etc.).
3. Add a soft reminder that this is not medical diagnosis and they should talk to a doctor for serious concerns.
Make it short, friendly, and easy to read.
"""

    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"LLM explanation is not available right now (Ollama error: {e})."


# ---------- Ollama helper (Q&A)
def call_ollama_qa(row: dict, question: str) -> str:
    """
    Free-form Q&A coach. Uses the current profile + question.
    """
    if not HAS_OLLAMA:
        return (
            "Q&A coach is not available because the 'ollama' Python package is not installed."
        )

    system_prompt = (
        "You are a friendly heart-health lifestyle coach. "
        "You answer in very simple English, step by step. "
        "You NEVER give strict medical diagnosis. "
        "You always say that people should talk to a doctor for serious issues."
    )

    summary_lines = [f"{k}: {v}" for k, v in row.items()]

    user_prompt = f"""
Here is the person's health profile:

{chr(10).join(summary_lines)}

The person is asking this question:

\"\"\"{question}\"\"\"\

Please answer:
- In simple, clear English
- With practical lifestyle tips
- No strict diagnosis
- End with a reminder to consult a doctor for medical advice.
"""

    try:
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        return f"Q&A coach is not available right now (Ollama error: {e})."


# ---------- Core inference ----------
def infer(
    age,
    gender,
    height,
    weight,
    bmi,
    sleep,
    steps,
    hr,
    bp_sys,
    bp_dia,
    ex_hrs,
    calories,
    alcohol,
    smoker,
    diabetic,
):

    if (bmi is None or bmi == "") and height and weight:
        bmi_val = weight / ((height / 100) ** 2)
    else:
        bmi_val = bmi

    row = {
        "Age": age,
        "Gender": gender,
        "Height_cm": height,
        "Weight_kg": weight,
        "BMI": bmi_val,
        "Hours_of_Sleep": sleep,
        "Daily_Steps": steps,
        "Heart_Rate": hr,
        "Systolic": bp_sys,
        "Diastolic": bp_dia,
        "Exercise_Hours_per_Week": ex_hrs,
        "Calories_Intake": calories,
        "Alcohol_Consumption_per_Week": alcohol,
        "Smoker": "Yes" if smoker == "Yes" else "No",
        "Diabetic": "Yes" if diabetic == "Yes" else "No",
    }

    df = pd.DataFrame([row])
    needed = list(num_features) + list(cat_features)
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    df = df[needed]

    # ----- ML prediction -----
    proba = pipe.predict_proba(df)[0]

    try:
        idx_yes = list(classes).index("Yes")
        idx_no = list(classes).index("No")
    except ValueError:
        idx_yes = 1 if len(classes) > 1 else 0
        idx_no = 0

    p_yes = float(proba[idx_yes])
    p_no = float(proba[idx_no])

    idx = int(np.argmax(proba))
    label = classes[idx]

    probs_md = (
        f"- **P(No) = {p_no*100:.1f}%**\n"
        f"- **P(Yes) = {p_yes*100:.1f}%**"
    )

    # ----- Rule-based lifestyle risk -----
    lifestyle_risk = rule_risk_level(row)

    pred_md = (
        "### 📊 Prediction (ML Model)\n\n"
        f"**Predicted class:** {label}\n\n"
        f"{probs_md}\n\n"
        "### ⚠️ Rule-based Lifestyle Risk\n\n"
        f"**Overall lifestyle risk: {lifestyle_risk}** "
        "(based on BMI, blood pressure, smoking, diabetes, sleep, steps, exercise, alcohol)."
    )

    # ----- Agent + Ontology -----
    recs, onto_notes = agent_suggestions(row)

    # ----- RAG -----
    query = (
        f"sleep {sleep} hours; exercise {ex_hrs} h/week; steps {steps}; "
        f"HR {hr}; BP {bp_sys}/{bp_dia}; BMI {bmi_val}; alcohol {alcohol}/wk; "
        f"smoker {row['Smoker']}; diabetic {row['Diabetic']}"
    )
    rag_passages = rag.query(query, k=3)

    tips = "### 💡 Personalized Suggestions\n\n"
    for r in recs:
        tips += f"- {r}\n"

    if onto_notes:
        tips += "\n### 🧬 Ontology Notes (Key Risk Factors)\n\n"
        for note in onto_notes:
            tips += f"{note}\n"

    tips += "\n---\n\n### 📚 Evidence-based Tips (RAG)\n\n"
    for p in rag_passages:
        tips += f"> {p}\n\n"

    # ----- Ollama explanation -----
    coach_text = call_ollama_coach(row, pred_md, tips)

    return pred_md, tips, coach_text


# ---------- CSV batch prediction ----------
def run_csv(file_path):
    df_in = pd.read_csv(file_path)
    needed = list(num_features) + list(cat_features)
    for col in needed:
        if col not in df_in.columns:
            df_in[col] = np.nan
    proba = pipe.predict_proba(df_in[needed])
    idx = np.argmax(proba, axis=1)
    labels = [classes[i] for i in idx]

    out = df_in.copy()
    out["predicted_risk"] = labels
    for i, cls in enumerate(classes):
        out[f"P({cls})%"] = (proba[:, i] * 100).round(1)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    out.to_csv(tmp.name, index=False)
    return out.head(100), tmp.name


# ---------- Wrapper for UI (manual prediction) ----------
def infer_ui(
    age,
    gender,
    height_cm,
    weight,
    sleep,
    steps,
    hr,
    bp_sys,
    bp_dia,
    ex_hrs,
    alcohol,
    smoker,
    diabetic,
):
    bmi = None         
    calories = 2200    

    return infer(
        age,
        gender,
        height_cm,
        weight,
        bmi,
        sleep,
        steps,
        hr,
        bp_sys,
        bp_dia,
        ex_hrs,
        calories,
        alcohol,
        smoker,
        diabetic,
    )


# ---------- Wrapper for UI (Q&A) ----------
def qa_ui(
    age,
    gender,
    height_cm,
    weight,
    sleep,
    steps,
    hr,
    bp_sys,
    bp_dia,
    ex_hrs,
    alcohol,
    smoker,
    diabetic,
    question,
):
    """Use current sliders + question to call Q&A coach."""
    if not question or question.strip() == "":
        return "Please type a question for the AI coach."

    bmi = None
    calories = 2200

    if (bmi is None or bmi == "") and height_cm and weight:
        bmi_val = weight / ((height_cm / 100) ** 2)
    else:
        bmi_val = bmi

    row = {
        "Age": age,
        "Gender": gender,
        "Height_cm": height_cm,
        "Weight_kg": weight,
        "BMI": bmi_val,
        "Hours_of_Sleep": sleep,
        "Daily_Steps": steps,
        "Heart_Rate": hr,
        "Systolic": bp_sys,
        "Diastolic": bp_dia,
        "Exercise_Hours_per_Week": ex_hrs,
        "Calories_Intake": calories,
        "Alcohol_Consumption_per_Week": alcohol,
        "Smoker": "Yes" if smoker == "Yes" else "No",
        "Diabetic": "Yes" if diabetic == "Yes" else "No",
    }

    return call_ollama_qa(row, question)


# ---------- Gradio UI ----------
theme = gr.themes.Soft(primary_hue="indigo")

with gr.Blocks(
    title="Heart & Wellness Coach",
    theme=theme,
    css=CSS,
) as demo:

    # TAB 1: Manual prediction
    with gr.Tab("🔮 Heart & Wellness Coach"):
        with gr.Column(elem_id="neon-card"):

            gr.Markdown("## 💓 Heart & Wellness Coach", elem_id="neon-title")

            with gr.Row(elem_id="neon-content-row"):

                # ========== LEFT PANEL ==========
                with gr.Column(elem_id="form-panel"):

                    with gr.Row(elem_id="form-grid-row-1"):
                        with gr.Column():
                            gr.Markdown("**Age :-**", elem_classes="form-label")
                            age = gr.Number(show_label=False, value=35)

                            gr.Markdown("**Gender :-**", elem_classes="form-label")
                            gender = gr.Dropdown(
                                choices=["Male", "Female", "Other"],
                                value="Male",
                                show_label=False,
                            )

                            gr.Markdown("**Height (cm) :-**", elem_classes="form-label")
                            height_cm = gr.Number(show_label=False, value=170)

                            gr.Markdown("**Weight (kg) :-**", elem_classes="form-label")
                            weight = gr.Number(show_label=False, value=70)

                            gr.Markdown("**Diabetic :-**", elem_classes="form-label")
                            diabetic = gr.Dropdown(
                                choices=["No", "Yes"],
                                value="No",
                                show_label=False,
                            )

                            gr.Markdown("**Hours of Sleep per Night :-**", elem_classes="form-label")
                            sleep = gr.Number(show_label=False, value=6)

                        with gr.Column():
                            gr.Markdown("**Resting Heart Rate (bpm) :-**", elem_classes="form-label")
                            hr = gr.Number(show_label=False, value=70)

                            gr.Markdown("**Systolic BP (mmHg) :-**", elem_classes="form-label")
                            bp_sys = gr.Number(show_label=False, value=120)

                            gr.Markdown("**Diastolic BP (mmHg) :-**", elem_classes="form-label")
                            bp_dia = gr.Number(show_label=False, value=82)

                            gr.Markdown("**Exercise Hours per Week :-**", elem_classes="form-label")
                            ex_hrs = gr.Number(show_label=False, value=1.5)

                            gr.Markdown("**Alcohol (drinks/week) :-**", elem_classes="form-label")
                            alcohol = gr.Number(show_label=False, value=2)

                            gr.Markdown("**Smoker :-**", elem_classes="form-label")
                            smoker = gr.Dropdown(
                                choices=["No", "Yes"],
                                value="No",
                                show_label=False,
                            )

                            gr.Markdown("**Steps per Day :-**", elem_classes="form-label")
                            steps = gr.Number(show_label=False, value=8000)

                    predict_btn = gr.Button("❤️ Predict Risk", elem_id="predict-btn")

                    # AI coach explanation (from prediction) – LEFT
                    with gr.Column(elem_id="ai-panel"):
                        gr.Markdown(
                            "### 🧠 AI Coach Explanation (Ollama LLM)",
                            elem_id="coach-heading",
                        )
                        coach_out = gr.Markdown(
                            "*A natural language explanation will appear here.*",
                            elem_id="coach-box",
                        )

                # ========== RIGHT PANEL ==========
                with gr.Column(elem_id="prediction-panel"):

                    # Prediction block
                    gr.Markdown("### 📊 Prediction", elem_id="pred-heading")
                    pred_out = gr.Markdown(
                        "*No prediction yet.*", elem_id="pred-box"
                    )

                    # Tips block
                    gr.Markdown(
                        "### 📚 Personalized Suggestions & Evidence-Based Tips",
                        elem_id="tips-heading",
                    )
                    tips_out = gr.Markdown(
                        "*Fill the details on the left and click **Predict Risk**.*",
                        elem_id="tips-box",
                    )

                    
                    with gr.Accordion(
                        "💬 Ask the AI Coach (Ollama LLM)",
                        open=False,
                        elem_id="qa-accordion",
                        
                    ):
                        gr.Markdown(
                            "You can ask follow-up questions about your heart health profile.",
                            elem_id="qa-heading",
                        )

                        gr.Markdown("**📝 Type your question**")
                        qa_question = gr.Textbox(
                            show_label=False,
                            placeholder="Example: How can I make my heart healthier?",
                            lines=3,
                        )

                        qa_btn = gr.Button("💡 Ask AI Coach", elem_id="qa-btn")

                        qa_answer = gr.Markdown(
                            "*The coach's answer will appear here.*",
                            elem_id="qa-box",
                        )

        # Click events
        predict_btn.click(
            fn=infer_ui,
            inputs=[
                age,
                gender,
                height_cm,
                weight,
                sleep,
                steps,
                hr,
                bp_sys,
                bp_dia,
                ex_hrs,
                alcohol,
                smoker,
                diabetic,
            ],
            outputs=[pred_out, tips_out, coach_out],
        )

        qa_btn.click(
            fn=qa_ui,
            inputs=[
                age,
                gender,
                height_cm,
                weight,
                sleep,
                steps,
                hr,
                bp_sys,
                bp_dia,
                ex_hrs,
                alcohol,
                smoker,
                diabetic,
                qa_question,
            ],
            outputs=[qa_answer],
        )

    # TAB 2: CSV batch
    with gr.Tab("📄 CSV Upload & Download"):
        gr.Markdown(
            "Upload a CSV with the same structure as **health_activity_data.csv** "
            "to generate batch predictions."
        )
        file_up = gr.File(label="Upload CSV", file_count="single", type="filepath")
        run_btn = gr.Button("Run Batch Predictions")
        table_out = gr.Dataframe(label="Preview (first 100 rows)")
        file_out = gr.File(label="Download Predictions CSV")

        run_btn.click(
            fn=run_csv,
            inputs=[file_up],
            outputs=[table_out, file_out],
        )

    gr.Markdown(
        "> **Disclaimer:** This is a student prototype and **not** medical advice. "
        "Always consult a qualified doctor for diagnosis or treatment.",
        elem_id="disclaimer",
    )

demo.queue().launch()


