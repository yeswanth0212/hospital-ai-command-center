from typing import List, Dict, Any
from backend.models import Observation, Action, ActionType

class Grader:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def score(self, obs_list: List[Observation], act_list: List[Action], info_list: List[Dict[str, Any]]) -> float:
        """Returns a deterministic score between 0.0 and 1.0"""
        raise NotImplementedError

class EasyGrader(Grader):
    def __init__(self):
        super().__init__("Prioritize High Severity Patients")

    def score(self, obs_list: List[Observation], act_list: List[Action], info_list: List[Dict[str, Any]]) -> float:
        correct_icu = 0
        total_icu_assigned = 0
        
        for i, action in enumerate(act_list):
            if action.action_type == ActionType.ICU:
                total_icu_assigned += 1
                explanation = info_list[i].get("reward_explanation", "")
                if "Correct ICU allocation" in explanation:
                    correct_icu += 1
                    
        if total_icu_assigned == 0:
            return 0.0
        return min(1.0, max(0.0, correct_icu / total_icu_assigned))

class MediumGrader(Grader):
    def __init__(self):
        super().__init__("Efficient Resource Allocation")

    def score(self, obs_list: List[Observation], act_list: List[Action], info_list: List[Dict[str, Any]]) -> float:
        correct_allocations = 0
        total_allocations = 0
        penalties = 0

        for i, action in enumerate(act_list):
            if action.action_type in [ActionType.ICU, ActionType.GENERAL]:
                total_allocations += 1
                explanation = info_list[i].get("reward_explanation", "")
                if "Correct" in explanation:
                    correct_allocations += 1
                elif "Penalty" in explanation or "Risk" in explanation:
                    penalties += 1

        if total_allocations == 0:
            return 0.0
            
        base_efficiency = correct_allocations / total_allocations
        penalty_reduction = (penalties / total_allocations) * 0.5
        
        score = base_efficiency - penalty_reduction
        return min(1.0, max(0.0, score))

class HardGrader(Grader):
    def __init__(self):
        super().__init__("Minimize Deaths & Maximize Throughput")

    def score(self, obs_list: List[Observation], act_list: List[Action], info_list: List[Dict[str, Any]]) -> float:
        if not obs_list:
            return 0.0
            
        final_obs = obs_list[-1]
        
        # Maximize survival rate directly
        survival_rate = final_obs.survival_rate / 100.0  # Convert to 0-1
        
        # Penalize for deaths directly against the score
        death_penalty = (final_obs.deaths * 0.1)
        
        # Maximize utilization 
        avg_utilization = sum(obs.resource_util for obs in obs_list) / len(obs_list) if obs_list else 0
        util_score = avg_utilization / 100.0
        
        # Combined objective: Survival is paramount, then throughput
        combined = (survival_rate * 0.7) + (util_score * 0.3) - death_penalty
        return min(1.0, max(0.0, combined))
