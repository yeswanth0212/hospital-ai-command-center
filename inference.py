import os
import requests
import json
from backend.models import Observation, Action, ActionType
from backend.graders import EasyGrader, MediumGrader, HardGrader
# Inference Agent handles direct LLM calls and heuristics
from backend.agent import InferenceAgent

def run_simulation_loop():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    agent = InferenceAgent(model_name=model_name, openai_api_key=openai_api_key)
    
    # Storage for grading
    obs_history = []
    act_history = []
    info_history = []
    
    print("[START] Simulation Episode")
    
    try:
        obs_res = requests.post(f"{api_base_url}/reset")
        obs_res.raise_for_status()
        obs_dict = obs_res.json()
        obs = Observation(**obs_dict)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to backend at {api_base_url}: {e}")
        return

    obs_history.append(obs)
    
    done = False
    while not done:
        # Agent calculates action
        action_dict = agent.get_action(obs_dict, mode=agent.mode)
        
        # Build strict payload
        payload = {
            "action_type": action_dict.get("action_type", "WAIT"),
            "patient_id": action_dict.get("patient_id", None)
        }
        
        print(f"[STEP] Action: {payload['action_type']}, Patient: {payload['patient_id']}")
        
        # Step environment
        step_res = requests.post(f"{api_base_url}/step", json=payload)
        step_data = step_res.json()
        
        obs_dict = step_data["observation"]
        obs = Observation(**obs_dict)
        act = Action(**payload)
        info = step_data.get("info", {})
        
        done = step_data["done"]
        
        obs_history.append(obs)
        act_history.append(act)
        info_history.append(info)

    print("[END] Episode Complete")
    print("--------------------------------------------------")
    print("EVALUATION RESULTS")
    print("--------------------------------------------------")
    
    # Calculate Grader Scores
    easy_grader = EasyGrader()
    medium_grader = MediumGrader()
    hard_grader = HardGrader()
    
    easy_score = easy_grader.score(obs_history, act_history, info_history)
    medium_score = medium_grader.score(obs_history, act_history, info_history)
    hard_score = hard_grader.score(obs_history, act_history, info_history)
    
    print(f"Task 1 (Easy) Score:   {easy_score:.2f} / 1.00")
    print(f"Task 2 (Medium) Score: {medium_score:.2f} / 1.00")
    print(f"Task 3 (Hard) Score:   {hard_score:.2f} / 1.00")

if __name__ == "__main__":
    run_simulation_loop()
