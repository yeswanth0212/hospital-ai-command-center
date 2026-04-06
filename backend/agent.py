import os
import json
import openai
from typing import List, Dict, Any, Optional
from backend.models import Action, ActionType

CONDITION_ICU_REASONS = {
    "cardiac": "continuous cardiac monitoring and emergency defibrillation required",
    "trauma": "hemorrhagic shock risk requires invasive hemodynamic support",
    "respiratory": "mechanical ventilation and O2 saturation control needed",
    "neurological": "ICP monitoring and neuro-critical care protocols required",
    "sepsis": "vasopressor therapy and continuous organ function monitoring required",
    "general": "multi-organ risk assessment requires intensive observation",
}

CONDITION_GENERAL_REASONS = {
    "cardiac": "stable cardiac rhythm allows standard bed monitoring protocol",
    "trauma": "wounds are controlled; recovery observation in general ward is sufficient",
    "respiratory": "oxygen via nasal cannula manageable in general ward setting",
    "neurological": "neurological status is stable; routine neuro-checks are adequate",
    "sepsis": "responding to antibiotics; step-down to general ward appropriate",
    "general": "clinical indicators do not require ICU-level resource consumption",
}


class InferenceAgent:
    def __init__(self, model_name="gpt-4o-mini", openai_api_key=None):
        self.model_name = model_name
        self.openai_api_key = openai_api_key or "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"
        if not self.openai_api_key:
            self.mode = "heuristic"
        else:
            self.mode = "llm"

    def get_action(self, observation: Dict[str, Any], mode: str = "llm") -> Dict[str, Any]:
        if mode == "heuristic" or self.mode == "heuristic":
            return self._heuristic_with_rationale(observation)

        # LLM-based action selection
        prompt = self._build_prompt(observation)
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Senior Hospital Triage Director. For EVERY allocation decision, "
                            "you MUST provide two distinct clinical justifications:\n"
                            "1. WHY this specific patient was selected over others in the queue (compare severity, condition urgency, wait time).\n"
                            "2. WHY this specific bed type (ICU/General) is the right medical choice for their condition.\n\n"
                            "Return ONLY a JSON object with these EXACT keys:\n"
                            '{"patient_id": "...", "patient_name": "...", "action_type": "ICU/GENERAL/WAIT", '
                            '"confidence": 0.XX, '
                            '"why_patient": "Reason this patient was selected over others", '
                            '"why_bed": "Reason this specific bed type is clinically appropriate"}'
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            decision = response.choices[0].message.content.strip()
            return self._parse_decision(decision, observation)
        except Exception as e:
            print(f"AI Decision Error: {e}")
            return self._heuristic_with_rationale(observation)

    def _heuristic_with_rationale(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        action = self._heuristic_action(observation)
        patients = observation.get("patients", [])
        waiting = [p for p in patients if (p["status"] if isinstance(p, dict) else p.status) == "WAITING"]
        waiting.sort(key=lambda x: (x["severity"] if isinstance(x, dict) else x.severity), reverse=True)

        patient_details = next(
            (p for p in patients if (p["id"] if isinstance(p, dict) else p.id) == action.patient_id), None
        )

        if patient_details:
            p_name = patient_details["name"] if isinstance(patient_details, dict) else patient_details.name
            p_sev = patient_details["severity"] if isinstance(patient_details, dict) else patient_details.severity
            p_cond = (patient_details["condition"] if isinstance(patient_details, dict) else patient_details.condition).lower()
            p_wait = patient_details.get("waiting_time", 0) if isinstance(patient_details, dict) else getattr(patient_details, "waiting_time", 0)

            # Build WHY PATIENT
            others = [p for p in waiting if (p["id"] if isinstance(p, dict) else p.id) != action.patient_id]
            if others:
                second_sev = others[0]["severity"] if isinstance(others[0], dict) else others[0].severity
                second_name = others[0]["name"] if isinstance(others[0], dict) else others[0].name
                why_patient = (
                    f"{p_name} has the highest critical score ({p_sev}/10) in the current queue, "
                    f"exceeding {second_name} ({second_sev}/10). "
                    f"Waiting time of {p_wait} steps further elevates mortality risk."
                )
            else:
                why_patient = f"{p_name} is the only patient in the queue (Severity: {p_sev}/10). Immediate allocation required."

            # Build WHY BED
            bed_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
            if "ICU" in bed_type:
                bed_reason = CONDITION_ICU_REASONS.get(p_cond, CONDITION_ICU_REASONS["general"])
                why_bed = f"ICU allocated because {bed_reason} for {p_cond} at severity {p_sev}."
            elif "GENERAL" in bed_type:
                bed_reason = CONDITION_GENERAL_REASONS.get(p_cond, CONDITION_GENERAL_REASONS["general"])
                why_bed = f"General Ward allocated because {bed_reason} for {p_cond} at severity {p_sev}."
            else:
                why_bed = "No beds available. Patient remains in queue — high-priority hold."

            return {
                "patient_id": action.patient_id,
                "patient_name": p_name,
                "action_type": bed_type,
                "confidence": 0.97,
                "why_patient": why_patient,
                "why_bed": why_bed,
                "explanation": f"{why_patient} | {why_bed}"
            }
        else:
            return {
                "patient_id": None,
                "patient_name": "SYSTEM",
                "action_type": "WAIT",
                "confidence": 1.0,
                "why_patient": "No patients are currently in the waiting queue.",
                "why_bed": "No bed allocation necessary at this time.",
                "explanation": "Queue is empty. Holding all resources."
            }

    def _heuristic_action(self, observation: Dict[str, Any]) -> Action:
        icu_available = observation.get("icu_available", 0)
        general_available = observation.get("general_available", 0)
        patients = observation.get("patients", [])

        waiting_patients = [p for p in patients if (p["status"] if isinstance(p, dict) else p.status) == "WAITING"]
        if not waiting_patients:
            return Action(action_type=ActionType.WAIT)

        waiting_patients.sort(key=lambda x: (x["severity"] if isinstance(x, dict) else x.severity), reverse=True)
        top_p = waiting_patients[0]
        p_id = top_p["id"] if isinstance(top_p, dict) else top_p.id
        p_sev = top_p["severity"] if isinstance(top_p, dict) else top_p.severity

        if p_sev >= 7:
            if icu_available > 0: return Action(patient_id=p_id, action_type=ActionType.ICU)
            elif general_available > 0: return Action(patient_id=p_id, action_type=ActionType.GENERAL)
        else:
            if general_available > 0: return Action(patient_id=p_id, action_type=ActionType.GENERAL)
            elif icu_available > 0: return Action(patient_id=p_id, action_type=ActionType.ICU)

        return Action(action_type=ActionType.WAIT)

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        waiting = [p for p in observation['patients'] if (p['status'] if isinstance(p, dict) else p.status) == "WAITING"]
        waiting.sort(key=lambda x: x['severity'] if isinstance(x, dict) else x.severity, reverse=True)
        prompt = f"Available beds — ICU: {observation['icu_available']}, General Ward: {observation['general_available']}\n\nPatient Queue (ranked by severity):\n"
        for p in waiting:
            pid = p['id'] if isinstance(p, dict) else p.id
            pname = p['name'] if isinstance(p, dict) else getattr(p, 'name', 'Unknown')
            sev = p['severity'] if isinstance(p, dict) else p.severity
            cond = p['condition'] if isinstance(p, dict) else p.condition
            wait = p.get('waiting_time', 0) if isinstance(p, dict) else getattr(p, 'waiting_time', 0)
            prompt += f"- ID: {pid}, Name: {pname}, Severity: {sev}/10, Condition: {cond}, Waiting: {wait} steps\n"
        prompt += "\nSelect the best patient and explain both why that patient AND why the bed type."
        return prompt

    def _parse_decision(self, decision: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        try:
            start = decision.find('{')
            end = decision.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(decision[start:end])
                # Ensure why_patient and why_bed exist
                if "why_patient" not in data:
                    data["why_patient"] = data.get("explanation", "Patient selected by AI.")
                if "why_bed" not in data:
                    data["why_bed"] = f"Bed type {data.get('action_type', 'WAIT')} selected by AI."
                if "explanation" not in data:
                    data["explanation"] = f"{data['why_patient']} | {data['why_bed']}"
                if "patient_name" not in data:
                    # Try to find name from observation
                    pid = data.get("patient_id")
                    p = next((x for x in observation.get("patients", []) if (x["id"] if isinstance(x, dict) else x.id) == pid), None)
                    data["patient_name"] = p["name"] if p and isinstance(p, dict) else (getattr(p, "name", "Unknown") if p else "Unknown")
                return data
        except Exception as ex:
            print(f"Parse error: {ex}")
        return self._heuristic_with_rationale(observation)
