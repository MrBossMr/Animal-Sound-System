"""
Intelligent Clinical Decision Support System (CDSS) - Animal Health Monitoring
Module: Clinical Recommendation Engine (Vantara Health & Care Protocols)
Generates dual-track actionable guidance:
1. Veterinary Clinical Interventions (for attending vets)
2. Caregiver Operational Action Plan (for animal keepers & shelter staff)
"""

from typing import Dict, List, Any


class ClinicalRecommendationEngine:
    """
    Generates evidence-based clinical and husbandry recommendations tailored to:
    - Species
    - Triage Urgency Level
    - Primary Diagnosed Condition
    - Acoustic Distress Profile
    """

    @staticmethod
    def generate_recommendations(species: str,
                                 condition: str,
                                 urgency: str,
                                 certainty_factor: float,
                                 confidence: float) -> Dict[str, Any]:
        """
        Synthesizes veterinary medical orders and caregiver husbandry protocols.
        """
        vet_orders: List[Dict[str, str]] = []
        caregiver_tasks: List[Dict[str, str]] = []
        monitoring_plan: Dict[str, Any] = {}

        if "Emergency" in urgency:
            monitoring_plan = {
                'check_frequency': 'Continuous monitoring / Every 15 minutes',
                'vitals_to_track': ['SpO2 / Mucous membrane color', 'Respiratory rate & effort', 'Heart rate / Pulse quality', 'Core body temperature'],
                'triage_color': '#d9534f',
                'status_tag': 'CRITICAL EMERGENCY'
            }

            vet_orders = [
                {
                    'category': 'Airway & Ventilation',
                    'action': 'Emergency Oxygenation',
                    'detail': f'Deliver 100% supplemental O2 via flow-by mask or oxygen cage immediately for the {species}.'
                },
                {
                    'category': 'Diagnostics',
                    'action': 'Point-of-Care Ultrasound (POCUS) & Thoracic Radiographs',
                    'detail': 'Perform emergency TFAST scan to rule out pneumothorax, pleural effusion, or pulmonary edema.'
                },
                {
                    'category': 'Pharmacology',
                    'action': 'Rapid-acting Bronchodilator / Corticosteroid',
                    'detail': 'Administer injectable bronchodilator (e.g., Terbutaline / Aminophylline) or Dexamethasone SP if inflammatory obstruction is present.'
                },
                {
                    'category': 'Vascular Access',
                    'action': 'Intravenous Catheterization',
                    'detail': 'Establish peripheral venous access for emergency shock rate crystalloid fluid therapy.'
                }
            ]

            caregiver_tasks = [
                {
                    'task': 'Immediate Quarantine / Isolation',
                    'detail': 'Move patient gently to a quiet, temperature-controlled triage room without stress or forceful restraint.'
                },
                {
                    'task': 'Clear Enclosure Hazards',
                    'detail': 'Remove hard obstacles, water troughs, or rough bedding that could cause trauma during recumbency.'
                },
                {
                    'task': 'Acoustic & Sensory Shielding',
                    'detail': 'Dim lighting and restrict all non-essential human traffic to minimize catecholamine-driven respiratory compromise.'
                }
            ]

        elif "Urgent" in urgency:
            monitoring_plan = {
                'check_frequency': 'Every 1 to 2 hours',
                'vitals_to_track': ['Pain score (Grimace scale)', 'Rectal temperature', 'Gastrointestinal motility / Rumination sounds', 'Fluid intake'],
                'triage_color': '#f0ad4e',
                'status_tag': 'URGENT MEDICAL ATTENTION'
            }

            vet_orders = [
                {
                    'category': 'Pain Management',
                    'action': 'Multimodal Analgesia Protocol',
                    'detail': f'Administer species-appropriate NSAID (e.g. Meloxicam or Flunixin meglumine) or opioid partial agonist for {species}.'
                },
                {
                    'category': 'Clinical Diagnostics',
                    'action': 'Comprehensive Blood Panel (CBC + Chemistry)',
                    'detail': 'Evaluate leukogram for acute inflammation, serum electrolytes, BUN, creatinine, and hepatic enzymes.'
                },
                {
                    'category': 'Abdominal / Musculoskeletal Scan',
                    'action': 'Focused Ultrasound & Palpation',
                    'detail': 'Examine abdomen for tympany, intestinal impaction, or musculoskeletal joint effusion.'
                }
            ]

            caregiver_tasks = [
                {
                    'task': 'Thermal Comfort Adjustment',
                    'detail': 'Ensure ambient enclosure temperature is maintained at species comfort zone (20-24°C for mammals/birds, controlled terrarium for amphibians).'
                },
                {
                    'task': 'Dietary Restriction',
                    'detail': 'Withhold bulky high-fiber concentrates until gastrointestinal tract clearance is confirmed by attending vet.'
                },
                {
                    'task': 'Behavioral Comfort Bedding',
                    'detail': 'Provide deep padded straw or orthopaedic foam bedding to relieve joint and abdominal pressure.'
                }
            ]

        elif "Moderate" in urgency:
            monitoring_plan = {
                'check_frequency': 'Every 4 to 6 hours',
                'vitals_to_track': ['Activity / Behavioral interaction', 'Feed & water consumption', 'Stool/elimination consistency'],
                'triage_color': '#5bc0de',
                'status_tag': 'MODERATE SURVEILLANCE'
            }

            vet_orders = [
                {
                    'category': 'Observation & Clinical Exam',
                    'action': 'Targeted Physical Examination',
                    'detail': f'Conduct routine wellness physical exam for {species} within 12 hours.'
                },
                {
                    'category': 'Diagnostics',
                    'action': 'Fecal Floatation & Urinalysis',
                    'detail': 'Screen for internal parasitism or metabolic subclinical stress markers.'
                }
            ]

            caregiver_tasks = [
                {
                    'task': 'Environmental Enrichment',
                    'detail': 'Introduce sensory enrichment, foraging toys, or visual dividers to reduce stress vocalization and stereotypic behaviors.'
                },
                {
                    'task': 'Conspecific Social Harmony',
                    'detail': 'Check herd or pack hierarchy to ensure the patient is not being bullied or isolated by dominant individuals.'
                }
            ]

        else: # Routine Monitoring
            monitoring_plan = {
                'check_frequency': 'Standard daily round (Every 12 to 24 hours)',
                'vitals_to_track': ['General demeanor and appetite', 'Normal vocalization patterns'],
                'triage_color': '#5cb85c',
                'status_tag': 'ROUTINE HEALTHY BASELINE'
            }

            vet_orders = [
                {
                    'category': 'Preventative Medicine',
                    'action': 'Routine Wellness Log',
                    'detail': f'Archive baseline vocalization acoustic signature for longitudinal health tracking of this {species}.'
                }
            ]

            caregiver_tasks = [
                {
                    'task': 'Standard Care Routine',
                    'detail': 'Maintain standard scheduled feeding, clean fresh water replenishment, and routine enclosure hygiene.'
                }
            ]

        return {
            'species': species,
            'condition': condition,
            'urgency': urgency,
            'certainty_factor': certainty_factor,
            'confidence': confidence,
            'monitoring_plan': monitoring_plan,
            'veterinary_orders': vet_orders,
            'caregiver_tasks': caregiver_tasks
        }
