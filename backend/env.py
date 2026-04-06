import random
import uuid
import math
from typing import List, Dict, Optional, Tuple, Any
from backend.models import Patient, PatientStatus, Action, ActionType, Observation, Reward, StepResult, Scenario

class HospitalEnv:
    def __init__(self, icu_capacity: int = 5, general_capacity: int = 15, max_steps: int = 50):
        self.icu_capacity = icu_capacity
        self.general_capacity = general_capacity
        self.max_steps = max_steps
        self.scenario = Scenario.NORMAL
        self.patient_names = [
            "John Smith", "Maria Johnson", "Robert Garcia", "Linda Brown", "Michael Davis", "Elizabeth Miller",
            "William Wilson", "Barbara Moore", "Richard Taylor", "Susan Anderson", "Joseph Thomas", "Jessica Jackson",
            "Thomas White", "Sarah Harris", "Charles Martin", "Karen Thompson", "Christopher Moore", "Nancy Young",
            "Paul King", "Alice Wright", "Bob Scott", "Carol Harris", "James Walker", "Betty Moore"
        ]
        self.reset()

    def set_scenario(self, scenario: Scenario):
        self.scenario = scenario
        if scenario == Scenario.OVERLOAD:
            self.icu_capacity = 3
            self.general_capacity = 10
        elif scenario == Scenario.SHORTAGE:
            self.icu_capacity = 2
            self.general_capacity = 5
        else:
            self.icu_capacity = 5
            self.general_capacity = 15

    def reset(self):
        self.patients: Dict[str, Patient] = {}
        self.icu_occupied = 0
        self.general_occupied = 0
        self.current_step = 0
        self.deaths = 0
        self.recovered = 0
        self.cumulative_reward = 0.0
        self.history: List[Dict] = []
        # Initial patients
        for _ in range(3):
            self._generate_patient()
        return self.get_observation()

    def _generate_patient(self):
        patient_id = str(uuid.uuid4())[:8]
        # Poisson-ish arrival could be simulated by calling this with a probability
        name = random.choice(self.patient_names)
        severity = random.randint(1, 10)
        condition = random.choice(["trauma", "cardiac", "general"])
        # Higher severity patients deteriorate faster if not treated
        deterioration_rate = 0.05 + (severity / 50.0)
        patient = Patient(
            id=patient_id,
            name=name,
            severity=severity,
            condition=condition,
            deterioration_rate=deterioration_rate
        )
        self.patients[patient_id] = patient

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        before_state = {
            "icu": self.icu_occupied,
            "gen": self.general_occupied,
            "wait": len([p for p in self.patients.values() if p.status == PatientStatus.WAITING])
        }
        self.current_step += 1
        reward_value = 0.0
        explanation = ""
        reason_points = []

        # Process Action
        if action.action_type == ActionType.ICU and action.patient_id:
            patient = self.patients.get(action.patient_id)
            if patient and patient.status == PatientStatus.WAITING:
                if self.icu_occupied < self.icu_capacity:
                    patient.status = PatientStatus.ALLOCATED_ICU
                    self.icu_occupied += 1
                    if patient.severity >= 7:
                        reward_value += 1.0
                        explanation = f"Correct ICU allocation for critical patient {patient.id}."
                        reason_points = ["Severity Index >= 7", "ICU Bed Available", "Optimal Clinical Matching"]
                    else:
                        reward_value -= 0.5
                        explanation = f"Allocated ICU bed to non-critical patient {patient.id} (Efficiency Penalty)."
                        reason_points = ["Severity Index < 7", "Low Priority for ICU", "Efficiency Loss Detected"]
                else:
                    reward_value -= 1.0
                    explanation = "No ICU beds available."

        elif action.action_type == ActionType.GENERAL and action.patient_id:
            patient = self.patients.get(action.patient_id)
            if patient and patient.status == PatientStatus.WAITING:
                if self.general_occupied < self.general_capacity:
                    patient.status = PatientStatus.ALLOCATED_GENERAL
                    self.general_occupied += 1
                    if patient.severity < 7:
                        reward_value += 0.5
                        explanation = f"Correct General bed allocation for patient {patient.id}."
                        reason_points = ["Moderate Severity", "Ward Slot Available", "Resource Optimization"]
                    else:
                        reward_value -= 1.0
                        explanation = f"Critical patient {patient.id} allocated to General bed (Risk Penalty)."
                        reason_points = ["Critical Patient in Ward", "High Clinical Risk", "Allocation Mismatch"]
                else:
                    reward_value -= 1.0
                    explanation = "No General beds available."

        elif action.action_type == ActionType.WAIT:
            reward_value -= 0.1
            explanation = "Timestep penalty for waiting."
            reason_points = ["System Wait Protocol", "Scanning New Arrivals", "Resource Preservation"]

        # Simulate Deterioration and Recovery
        for p_id, patient in list(self.patients.items()):
            if patient.status == PatientStatus.WAITING:
                patient.waiting_time += 1
                # Waiting patients deteriorate
                patient.severity += 1 if random.random() < patient.deterioration_rate else 0
                if patient.severity > 10:
                    patient.status = PatientStatus.DECEASED
                    self.deaths += 1
                    reward_value -= 2.0
                    explanation = f"Patient {patient.id} died while waiting."
            
            elif patient.status == PatientStatus.ALLOCATED_ICU:
                # ICU patients recover or deteriorate less
                if random.random() < 0.2: # Recovery chance
                    patient.severity -= 1
                if patient.severity <= 0:
                    patient.status = PatientStatus.RECOVERED
                    self.recovered += 1
                    self.icu_occupied -= 1
                    reward_value += 1.0
                    explanation = f"Patient {patient.id} recovered in ICU!"
                    del self.patients[p_id]
                elif patient.severity > 10: # Still a small risk
                    patient.status = PatientStatus.DECEASED
                    self.deaths += 1
                    self.icu_occupied -= 1
                    reward_value -= 2.0
                    del self.patients[p_id]

            elif patient.status == PatientStatus.ALLOCATED_GENERAL:
                # General patients recover slowly
                if random.random() < 0.1:
                    patient.severity -= 1
                # Risk of deterioration still exists if not enough care
                if random.random() < 0.05:
                    patient.severity += 1
                
                if patient.severity <= 0:
                    patient.status = PatientStatus.RECOVERED
                    self.recovered += 1
                    self.general_occupied -= 1
                    reward_value += 0.5
                    explanation = f"Patient {patient.id} recovered in General ward!"
                    del self.patients[p_id]
                elif patient.severity > 10:
                    patient.status = PatientStatus.DECEASED
                    self.deaths += 1
                    self.general_occupied -= 1
                    reward_value -= 2.0
                    del self.patients[p_id]

        # New Patient Arrival (Poisson-ish)
        if random.random() < 0.4: # 40% chance each step
            self._generate_patient()

        self.cumulative_reward += reward_value
        obs = self.get_observation()
        after_state = {
            "icu": self.icu_occupied,
            "gen": self.general_occupied,
            "wait": len([p for p in self.patients.values() if p.status == PatientStatus.WAITING])
        }
        done = self.deaths >= 5 or self.current_step >= self.max_steps 
        info = {
            "reward_explanation": explanation,
            "reason_points": reason_points,
            "before_state": before_state,
            "after_state": after_state
        }
        return obs, reward_value, done, info

    def get_observation(self) -> Observation:
        total_handled = self.recovered + self.deaths
        survival_rate = (self.recovered / max(1, total_handled)) * 100.0
        
        total_capacity = self.icu_capacity + self.general_capacity
        resource_util = int(((self.icu_occupied + self.general_occupied) / total_capacity) * 100) if total_capacity > 0 else 0
        
        # EFFICIENCY SCORE: (Allocations - Deaths) normalized
        efficiency_score = max(0, min(100, (survival_rate * 0.7 + resource_util * 0.3)))
        
        waiting_patients = [p for p in self.patients.values() if p.status == PatientStatus.WAITING]
        queue_length = len(waiting_patients)
        avg_wait_time = sum(p.waiting_time for p in waiting_patients) / max(1, queue_length)

        return Observation(
            icu_available=self.icu_capacity - self.icu_occupied,
            general_available=self.general_capacity - self.general_occupied,
            patients=list(self.patients.values()),
            current_step=self.current_step,
            cumulative_reward=self.cumulative_reward,
            deaths=self.deaths,
            survival_rate=round(survival_rate, 1),
            resource_util=resource_util,
            fairness_index=1.0,
            efficiency_score=round(efficiency_score, 1),
            scenario=self.scenario,
            avg_wait_time=round(avg_wait_time, 2),
            queue_length=queue_length
        )

    def state(self) -> Dict:
        return self.get_observation().model_dump()
