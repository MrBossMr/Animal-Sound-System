# Intelligent Clinical Decision Support System (CDSS)
## System Integration & Deployment Guide (Practical 9)
**Project Title:** Intelligent Decision Support System for Animal Health Monitoring and Care Recommendation using Vocalization Analysis  
**Laboratory:** Computational Intelligence Laboratory (CIL)  
**Context:** Inspired by Vantara Wildlife Health & Rehabilitation Infrastructure

---

## 1. System Architecture & Integration Workflow

Task 9 integrates all Computational Intelligence modules developed in Practicals 1 through 8 into a single unified Clinical Decision Support System (CDSS).

```
        INTELLIGENT CLINICAL DECISION SUPPORT SYSTEM (CDSS)
                                │
                                ▼
                       Patient Information
                                │
                                ▼
                        Input Validation
                                │
                                ▼
                 Feature Engineering (20 Acoustic Features)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
       Rule-Based Expert                     ML / DL Models
        System (Task 2)                            │
              │                                    ▼
              │                           Disease Prediction
              │                                    │
              └─────────────────┬─────────────────┘
                                ▼
                         XAI Explanation
                                │
                                ▼
                      Recommendation Engine
                                │
                                ▼
                        Clinical Dashboard
                                │
                                ▼
                     Generate Clinical Report
```

### Module Mapping Across Practicals

| Module | Source Practical | Description | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **Knowledge Representation** | **Practical 1** | Clinical entities, attributes, and Vantara ontology | Patient & Enclosure Schema |
| **Rule-Based Expert System** | **Practical 2** | Production rules, triage urgency, MYCIN certainty factors ($CF$) | `expert_system.py` |
| **State-Space Search** | **Practical 3** | Heuristic candidate state search (A*) | Triage path selection |
| **Data Cleaning & Exploration** | **Practical 4** | 200 animal audio files across 5 classes (Dog, Rooster, Pig, Cow, Frog) | Clean dataset in `Data/Data/` |
| **Feature Engineering** | **Practical 5** | Exact 20 acoustic features (MFCC, Spectral Contrast, Flatness, Duration) | `feature_engineering.py` |
| **Machine Learning** | **Practical 6** | Random Forest, SVM (RBF), k-NN, Soft-Voting Ensemble | `ml_model.py` |
| **Deep Learning** | **Practical 7** | 4-Layer DNN (128-64-32-5), 13,189 params, 87.50% test accuracy | `dl_model.py` |
| **Explainable AI (XAI)** | **Practical 8** | SHAP, LIME, Permutation Importance, Counterfactuals, Trust Score | `xai_module.py` |
| **System Integration & UI** | **Practical 9** | Full Streamlit Application, Clinical Dashboard & Report Generator | `app.py`, `report_generator.py` |

---

## 2. Directory Structure

```
CIL/
│
├── CDSS/
│   ├── app.py                     # Streamlit Main Dashboard (8 Interactive Tabs)
│   ├── expert_system.py           # Forward-Chaining Clinical Rule Engine
│   ├── ml_model.py                # Random Forest, SVM, k-NN & Ensemble Classifier
│   ├── dl_model.py                # 4-Layer Deep Neural Network Implementation
│   ├── feature_engineering.py     # 20 Acoustic Features Extractor & Validator
│   ├── xai_module.py              # Permutation Importance, SHAP, LIME & Dashboard
│   ├── recommendation.py          # Veterinary Orders & Caregiver Protocol Generator
│   ├── report_generator.py        # Print-ready HTML, Markdown & JSON Exporter
│   ├── utils.py                   # Bioacoustic Waveform/Spectrogram Plotting
│   ├── requirements.txt           # Project Dependencies
│   ├── DEPLOYMENT.md              # Deployment Manual & Architecture Guide
│   │
│   ├── data/
│   │   ├── feature_matrix_full.csv # 200 samples × 20 acoustic features
│   │   └── selected_features.csv   # Feature descriptions and metadata
│   │
│   └── models/
│       └── animal_sound_class_mapping.json # 5-class target dictionary
│
├── Data/Data/                     # 200 Raw Audio Files (.ogg) in 5 Subfolders
│   ├── 101 - Dog/
│   ├── 102 - Rooster/
│   ├── 103 - Pig/
│   ├── 104 - Cow/
│   └── 105 - Frog/
│
├── instruction.md                 # Lab Manual & Viva Specifications
└── Prac_7&8_CIL.ipynb             # Research Notebook for Practicals 7 & 8
```

---

## 3. Local Installation & Launch Guide

### Step 1: Open Terminal / PowerShell
Navigate to the `CDSS` directory:
```bash
cd "C:\Users\admin\Desktop\CIL\CDSS"
```

### Step 2: (Optional but Recommended) Create Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```

The application will start immediately and output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## 4. Key Functional Features of the CDSS Prototype

1. **Multimodal Clinical Input**:
   - Accepts animal demographic metadata (Species, Enclosure ID, Posture, Respiration, Appetite, Temperature, Audible Distress).
   - Audio input via 3 modalities: Real preset audio recordings from dataset, user-uploaded audio files (`.ogg`, `.wav`), or simulated acoustic features.
2. **Clinical Signal Validation**:
   - Verifies file format, signal duration, amplitude ceiling, and noise floor.
   - Detects acoustic outliers exceeding 4 standard deviations from reference baseline.
3. **Dual Decision Path (Cross-Modal Verification)**:
   - **Path A**: Rule-based expert system evaluates physical vitals and triage urgency using MYCIN certainty factors ($CF$).
   - **Path B**: 4-Layer DNN and Soft-Voting Ensemble classify bioacoustic phonations across the 5 animal classes with calibrated confidence.
   - **Concordance Verification**: Verifies whether acoustic predictions match the species profile and alert triage status.
4. **Comprehensive Explainable AI (XAI)**:
   - Visualizes the 4-panel Explainability Dashboard (Figure 5 from Practical 8).
   - Audits global SHAP values, local patient-specific waterfall contributions, and LIME local linear approximations.
   - Confirms counterfactual stability across $[-2.0\sigma, +2.0\sigma]$ range.
5. **Actionable Care Protocols**:
   - Detailed veterinary medical orders (Airway, POCUS ultrasound, analgesia, IV fluids).
   - Caregiver husbandry tasks (isolation, thermal regulation, bedding, feed modification).
6. **Automated Clinical Report Generator**:
   - One-click export to print-ready HTML, concise Markdown summary, and EHR/EMR JSON payload.

---

## 5. Viva Exam Preparation & Defense Guide

### Q1: What are the 20 acoustic features used as input to your DNN?
> **Answer:** Our DNN uses 20 acoustic features selected in Task 5: mainly **MFCC** coefficients (representing vocal tract resonance and timbre), **Spectral Contrast** across 7 sub-bands (measuring peak-to-valley frequency energy distribution), **Spectral Flatness** (indicating tone-like vs. noise-like harshness), and **Duration**.

### Q2: How do you explain the Training vs. Validation Accuracy curve (Practical 7)?
> **Answer:** The curve shows progressive learning across 50 epochs. Training accuracy increases from 26% to ~96%, while validation accuracy rises from 41% and stabilizes at 87.50%. The gap between the curves in later epochs represents a generalization gap, which we controlled using Dropout (0.30 and 0.20) and Early Stopping.

### Q3: How do you explain the Confusion Matrix?
> **Answer:** On the 40-sample test set (8 samples per class), our model correctly classified 35 samples, achieving 87.50% test accuracy. Correct classifications form the diagonal (Dog: 7, Rooster: 8, Pig: 7, Cow: 6, Frog: 7). Rooster vocalizations achieved 100% precision due to their distinct high-frequency harmonic structure.

### Q4: What is the clinical value of Explainable AI (Practical 8)?
> **Answer:** Black-box predictions cannot be safely deployed in veterinary medicine. XAI provides clinicians with auditability: SHAP confirms that predictions rely on anatomically valid spectral contrast and MFCC timbral bands rather than background noise; LIME provides local decision rules for individual cases; and counterfactual analysis proves that the acoustic prediction is robust against spurious audio spikes.
