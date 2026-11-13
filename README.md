
## Machine Learning Pipeline & Model Evaluation

The system implements a structured supervised learning pipeline for cardiovascular risk classification using a Random Forest model.

### Data Preprocessing & Feature Engineering
- **Missing Value Handling:** Data cleaning and consistency checks.
- **Blood Pressure Processing:** Separation of systolic and diastolic measurements.
- **BMI Computation:** Derived from height and weight.
- **Feature Transformation:** Numeric scaling, categorical encoding and data type conversion.
- **Metadata Export:** Feature information for consistent model inputs.

### Model Training & Evaluation

| Component | Technique / Parameter |
|---|---|
| ML Algorithm | Random Forest Classifier |
| Learning Approach | Supervised Binary Classification |
| Train/Test Split | 80% Training / 20% Testing |
| Evaluation Metrics | Accuracy, F1-Macro, ROC-AUC |
| Model Output | Heart-Risk Classification (Yes/No) |
| Probability Estimation | P(Yes) and P(No) |

<img width="940" height="657" alt="image" src="https://github.com/user-attachments/assets/65445091-bb11-4d5c-b8a4-e75e6a68f8ae" />
<img width="940" height="587" alt="image" src="https://github.com/user-attachments/assets/1868de83-493e-4c43-9362-b2ef1b78a7a6" />
<img width="437" height="622" alt="image" src="https://github.com/user-attachments/assets/f1f3170c-ca85-42e9-ac71-c67623998c70" />
<img width="414" height="621" alt="image" src="https://github.com/user-attachments/assets/40a3b6a5-8102-4552-bbe4-043a535003eb" />

## Agentic AI, RAG & LLM Architecture

The system integrates Machine Learning, rule-based reasoning, ontology-based knowledge representation, Retrieval-Augmented Generation (RAG), and locally hosted Generative AI.

### AI Workflow

**1. User Input**
- Health and lifestyle measurements
- Heart-risk prediction request
- Optional conversational follow-up questions

**2. Structured Data Preparation**
- Convert user inputs into structured data
- Generate summaries for LLM context
- Integrate predictions and lifestyle factors

**3. LLM Prompt Engineering**
- ML prediction results and probability estimates
- RAG-retrieved health guideline evidence
- Ontology-based risk factor explanations
- Rule-based lifestyle recommendations

**4. Ollama-Powered Local LLM**
- **LLM:** Llama 3.2 (3B)
- **Platform:** Ollama
- **Technique:** Context-Aware Prompt Construction
- **Outputs:** Natural-language explanations, personalised wellness insights and conversational Q&A

### RAG & Knowledge Retrieval

- **Retrieval Technique:** TF-IDF Vectorization
- **Similarity Measurement:** Cosine Similarity
- **Knowledge Source:** Health and lifestyle guidelines
- **Reasoning:** Rule-Based Agents + Health Ontology
- **Purpose:** Evidence-supported recommendations and reduced reliance on unsupported LLM responses

## End-to-End System Architecture

```text
          HEALTH & LIFESTYLE DATA
                     |
                     v
             DATA PREPROCESSING
       Missing Values | BMI | BP Split
                     |
                     v
             FEATURE ENGINEERING
        Scaling | Encoding | Metadata
                     |
                     v
         RANDOM FOREST CLASSIFIER
             80% Train / 20% Test
                     |
                     v
              MODEL EVALUATION
        Accuracy | F1-Macro | ROC-AUC
                     |
                     v
             ML RISK PREDICTION
                P(Yes) / P(No)
                     |
           +---------+---------+
           |                   |
           v                   v
     RULE-BASED AGENTS    RAG RETRIEVAL
     & HEALTH ONTOLOGY    TF-IDF / Cosine
           |                   |
           +---------+---------+
                     |
                     v
              PROMPT ENGINEERING
       ML + RAG + Ontology + Rules
                     |
                     v
          OLLAMA / LLAMA 3.2 (3B)
                     |
                     v
       PERSONALISED AI EXPLANATIONS
             & INTERACTIVE Q&A
                     |
                     v
              GRADIO INTERFACE
```

