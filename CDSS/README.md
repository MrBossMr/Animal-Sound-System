# Intelligent Clinical Decision Support System (CDSS)
## Animal Health Monitoring and Care Recommendation using Vocalization Analysis
**Computational Intelligence Laboratory (CIL) - Practical 9 Integration**

### Quick Start
```bash
# 1. Navigate to CDSS directory
cd C:\Users\admin\Desktop\CIL\CDSS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application on Localhost
streamlit run app.py
```

### Integrated Modules
- **`app.py`**: Streamlit 8-tab clinical web interface.
- **`feature_engineering.py`**: Input validation & extraction of 20 acoustic features (Task 5).
- **`expert_system.py`**: Forward-chaining rule engine with MYCIN Certainty Factors (Task 2).
- **`ml_model.py`**: Random Forest, SVM, k-NN, and soft-voting ensemble (Task 6).
- **`dl_model.py`**: 4-layer Deep Neural Network (13,189 parameters, 87.5% accuracy) (Task 7).
- **`xai_module.py`**: SHAP, LIME, permutation importance, counterfactuals, and Figure 5 dashboard (Task 8).
- **`recommendation.py`**: Dual veterinary medical orders and caregiver husbandry protocols.
- **`report_generator.py`**: Export clinical reports to HTML, Markdown, and EHR/EMR JSON.
- **`utils.py`**: Bioacoustic spectrograms, waveforms, and viva exam references.
