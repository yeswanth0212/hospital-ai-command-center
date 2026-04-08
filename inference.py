import os
import requests
from backend.models import Observation, Action
from backend.graders import EasyGrader, MediumGrader, HardGrader
from backend.agent import InferenceAgent


def run_simulation_loop():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # ✅ FIXED: no openai_api_key
    agent = InferenceAgent(model_name=model_name)

    obs_history = []
    act_history = []
    info_history = []

    print("[START] Simulation Episode")

    try:
        res = requests.post(f"{api_base_url}/reset")
        res.raise_for_status()
        obs_dict = res.json()
        obs = Observation(**obs_dict)
    except Exception as e:
        print("❌ Backend error:", e)
        return

    obs_history.append(obs)
    done = False

    while not done:
        try:
            action_dict = agent.get_action(obs_dict, mode="llm")

            payload = {
                "action_type": str(action_dict.get("action_type", "WAIT")),
                "patient_id": action_dict.get("patient_id")
            }

            print(f"[STEP] {payload}")

            step_res = requests.post(f"{api_base_url}/step", json=payload)

            if step_res.status_code != 200:
                print("⚠️ Step error:", step_res.text)
                break

            step_data = step_res.json()

            obs_dict = step_data["observation"]
            obs = Observation(**obs_dict)

            act = Action(**payload)
            info = step_data.get("info", {})
            done = step_data.get("done", True)

            obs_history.append(obs)
            act_history.append(act)
            info_history.append(info)

        except Exception as e:
            print("❌ Runtime error:", e)
            break

    print("[END] Episode Complete")

    try:
        print("Easy:", EasyGrader().score(obs_history, act_history, info_history))
        print("Medium:", MediumGrader().score(obs_history, act_history, info_history))
        print("Hard:", HardGrader().score(obs_history, act_history, info_history))
    except Exception as e:
        print("⚠️ Grading error:", e)


if __name__ == "__main__":
    run_simulation_loop()
