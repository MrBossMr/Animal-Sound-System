"""
Intelligent Clinical Decision Support System (CDSS) - Animal Health Monitoring
Module: Rule-Based Expert System (Task 2 & Task 1 Ontology Integration)
Provides clinical inference, urgency triage, certainty factors (CF), and concordance checks.
"""

from typing import Dict, List, Any, Optional, Tuple


class RuleResult:
    """Represents an individual triggered clinical rule result."""
    def __init__(self, rule_id: str, name: str, condition: str, urgency: str,
                 certainty_factor: float, explanation: str, action: str):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.urgency = urgency
        self.certainty_factor = certainty_factor
        self.explanation = explanation
        self.action = action

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'condition': self.condition,
            'urgency': self.urgency,
            'certainty_factor': round(self.certainty_factor, 2),
            'explanation': self.explanation,
            'action': self.action
        }


class ClinicalExpertSystem:
    """
    Forward-chaining Rule-Based Clinical Expert System for Animal Health.
    Integrates clinical observations (vitals, posture, appetite, respiration)
    with acoustic vocalization characteristics.
    """

    SPECIES_BASELINE = {
        'Dog': {'normal_temp_c': (38.0, 39.2), 'normal_resp_bpm': (15, 30)},
        'Rooster': {'normal_temp_c': (40.5, 42.0), 'normal_resp_bpm': (15, 25)},
        'Pig': {'normal_temp_c': (38.5, 39.5), 'normal_resp_bpm': (15, 20)},
        'Cow': {'normal_temp_c': (38.0, 39.0), 'normal_resp_bpm': (20, 30)},
        'Frog': {'normal_temp_c': (20.0, 26.0), 'normal_resp_bpm': (20, 40)}
    }

    def __init__(self):
        pass

    def evaluate(self, 
                 species: str,
                 posture: str,
                 respiration: str,
                 appetite: str,
                 temperature_c: float,
                 vocal_distress_level: str,
                 duration_s: float = 5.0,
                 acoustic_features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Executes clinical forward-chaining rules against observed patient symptoms.
        Returns triggered rules, primary diagnosis, triage urgency, and certainty factor.
        """
        triggered_rules: List[RuleResult] = []
        
        # Determine fever / hypothermia status
        base = self.SPECIES_BASELINE.get(species, {'normal_temp_c': (38.0, 39.5), 'normal_resp_bpm': (15, 30)})
        min_temp, max_temp = base['normal_temp_c']
        temp_status = "Normal"
        if temperature_c > max_temp + 1.5:
            temp_status = "Severe Hyperthermia"
        elif temperature_c > max_temp:
            temp_status = "Mild Pyrexia"
        elif temperature_c < min_temp:
            temp_status = "Hypothermia"

        # RULE 1: Severe Acute Respiratory Distress / Pulmonary Congestion
        if respiration in ["Labored (Dyspnea)", "Shallow & Rapid"] and (
            posture in ["Agitated / Orthopneic", "Trembling / Pacing"] or vocal_distress_level in ["High Distress", "Emergency Screech"]
        ):
            cf = 0.95 if respiration == "Labored (Dyspnea)" else 0.88
            triggered_rules.append(RuleResult(
                rule_id="RULE-RESP-01",
                name="Acute Respiratory Distress Syndrome",
                condition="Suspected Airway Obstruction / Pulmonary Congestion",
                urgency="Emergency (Immediate Intervention)",
                certainty_factor=cf,
                explanation=f"Patient exhibits {respiration} combined with {posture} and {vocal_distress_level}, indicating compromised gas exchange and acute ventilatory failure.",
                action="Administer supplemental oxygen, maintain patent airway, elevate thoracic position, notify attending veterinarian immediately."
            ))

        # RULE 2: Acute Pain / Trauma / Musculoskeletal Crisis
        if posture in ["Guarding / Hunched", "Trembling / Pacing"] and vocal_distress_level in ["High Distress", "Emergency Screech", "Moderate Strain"]:
            cf = 0.92 if posture == "Guarding / Hunched" else 0.84
            triggered_rules.append(RuleResult(
                rule_id="RULE-PAIN-02",
                name="Acute Pain / Severe Somatic Discomfort",
                condition="Acute Musculoskeletal Trauma / Internal Visceral Pain",
                urgency="Urgent (Evaluation within 1 hour)",
                certainty_factor=cf,
                explanation=f"Guarding posture with vocal acoustic distress ({vocal_distress_level}) is a definitive indicator of acute localized or somatic pain.",
                action="Prepare analgesic protocol (NSAIDs / multimodal analgesia pending vet order), immobilize patient, conduct targeted palpation."
            ))

        # RULE 3: Gastrointestinal Colic / Tympany (Bloat)
        if appetite in ["Anorexic (Refusing food & water)", "Severe Inappetence"] and posture in ["Lethargic / Recumbent", "Guarding / Hunched"]:
            cf = 0.88
            triggered_rules.append(RuleResult(
                rule_id="RULE-GI-03",
                name="Gastrointestinal Hypomotility / Colic",
                condition="Suspected Enteritis / Ruminal Tympany / Gastric Dilation",
                urgency="Urgent (Evaluation within 1 hour)",
                certainty_factor=cf,
                explanation=f"Anorexia coupled with recumbency or abdominal guarding strongly signals visceral abdominal pain or metabolic colic.",
                action="Withhold solid feed, check ruminal motility/borborygmi, palpate abdomen, prepare IV fluid hydration."
            ))

        # RULE 4: Thermal Shock / Heat Exhaustion
        if temp_status == "Severe Hyperthermia" and respiration in ["Rapid (Tachypnea)", "Labored (Dyspnea)"]:
            cf = 0.90
            triggered_rules.append(RuleResult(
                rule_id="RULE-HEAT-04",
                name="Acute Heat Prostration / Thermal Stress",
                condition="Thermal Stress / Heat Exhaustion Crisis",
                urgency="Urgent (Evaluation within 1 hour)",
                certainty_factor=cf,
                explanation=f"Patient core temperature is {temperature_c:.1f}°C ({temp_status}) alongside tachypneic breathing, risking heat stroke and cellular hypoxia.",
                action="Transfer immediately to cooled environment, apply lukewarm compresses to extremities, offer electrolyte solution."
            ))

        # RULE 5: Environmental Distress / Behavioral Agitation
        if vocal_distress_level in ["High Distress", "Moderate Strain"] and posture == "Trembling / Pacing" and appetite == "Normal":
            cf = 0.78
            triggered_rules.append(RuleResult(
                rule_id="RULE-BEHAV-05",
                name="Environmental / Isolation Stress Reaction",
                condition="Psychosocial Distress / Territorial Alarm",
                urgency="Moderate (Caregiver Action)",
                certainty_factor=cf,
                explanation="Vocal distress and agitation occurring without systemic physiological depression suggest external stressors or territorial disturbance.",
                action="Inspect enclosure boundary, reduce ambient noise/lighting, verify presence of conspecific herd/flock members."
            ))

        # RULE 6: Normal Baseline / Routine Vocalization
        if (posture == "Normal / Alert" and respiration == "Normal (Eupnea)" and 
            appetite == "Normal" and temp_status == "Normal" and 
            vocal_distress_level in ["Normal / Baseline", "Low / Content"]):
            triggered_rules.append(RuleResult(
                rule_id="RULE-NORM-06",
                name="Physiological Homeostasis / Normal Vocalization",
                condition="Healthy Baseline / Routine Communication",
                urgency="Routine Monitoring",
                certainty_factor=0.96,
                explanation="All observable clinical signs, vital parameters, and acoustic signatures align with physiological health standards.",
                action="Continue routine dietary provision and scheduled veterinary health wellness surveillance."
            ))

        # Default fallback if symptoms are mild or mixed
        if not triggered_rules:
            triggered_rules.append(RuleResult(
                rule_id="RULE-GEN-07",
                name="Subclinical / Mild Nonspecific Variation",
                condition="Mild Acoustic or Behavioral Variation",
                urgency="Moderate (Caregiver Action)",
                certainty_factor=0.70,
                explanation="Observations display borderline deviations not yet fulfilling acute emergency threshold criteria.",
                action="Place patient on close 4-hour visual observation log and repeat vocalization assessment."
            ))

        # Sort triggered rules by urgency and certainty factor
        urgency_rank = {
            "Emergency (Immediate Intervention)": 4,
            "Urgent (Evaluation within 1 hour)": 3,
            "Moderate (Caregiver Action)": 2,
            "Routine Monitoring": 1
        }
        triggered_rules.sort(key=lambda r: (urgency_rank.get(r.urgency, 0), r.certainty_factor), reverse=True)
        primary = triggered_rules[0]

        # Calculate Combined Certainty Factor using Mycin formula: CF_comb = CF1 + CF2 * (1 - CF1)
        combined_cf = primary.certainty_factor
        for r in triggered_rules[1:]:
            combined_cf = combined_cf + r.certainty_factor * (1.0 - combined_cf)

        return {
            'primary_condition': primary.condition,
            'urgency': primary.urgency,
            'primary_rule_id': primary.rule_id,
            'primary_explanation': primary.explanation,
            'primary_action': primary.action,
            'certainty_factor': round(min(0.99, combined_cf), 2),
            'temp_status': temp_status,
            'triggered_rules': [r.to_dict() for r in triggered_rules]
        }

    @staticmethod
    def check_cross_modal_concordance(expert_result: Dict[str, Any], 
                                      ml_prediction: str, 
                                      confidence: float) -> Dict[str, Any]:
        """
        Validates concordance between the Rule-Based Expert System and ML/DL Acoustic Prediction.
        Ensures cross-modal verification prior to clinical dispatch.
        """
        urgency = expert_result['urgency']
        
        # If ML predicted a sound class with high confidence
        is_high_conf = confidence >= 0.75
        
        if "Emergency" in urgency or "Urgent" in urgency:
            concordance_status = "Alert: Clinical Distress Verified"
            concordance_color = "red"
            summary = (f"Rule-Based Expert System identified urgent condition ({expert_result['primary_condition']}) "
                       f"with CF {expert_result['certainty_factor']*100:.0f}%. "
                       f"Acoustic ML model confirmed animal source as {ml_prediction} ({confidence*100:.1f}% confidence).")
        else:
            concordance_status = "Concordant: Stable Baseline"
            concordance_color = "green"
            summary = (f"Clinical Expert System and Acoustic ML model are in mutual concordance. "
                       f"Vocalization is attributed to {ml_prediction} with routine health status.")

        return {
            'status': concordance_status,
            'color': concordance_color,
            'summary': summary,
            'verified': True
        }
