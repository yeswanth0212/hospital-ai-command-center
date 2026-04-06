import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.env import HospitalEnv
from backend.models import Action, Observation, Reward, ActionType, StepResult, Scenario
from backend.agent import InferenceAgent
from typing import Dict, List, Optional, Any
import json

app = FastAPI(title="Smart Hospital AI Command Center API v4.0")

# Enable CORS for the React/Vanilla frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global simulation instance
env = HospitalEnv()
agent = InferenceAgent()

# Serve Frontend
app.mount("/dashboard", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def read_root():
    return {"message": "Final winning-level AI Backend is active (Pydantic V2)."}

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs

@app.post("/step", response_model=StepResult)
def step_env(action: Action):
    obs, reward_val, done, info = env.step(action)
    reward = Reward(
        value=reward_val, 
        explanation=info.get("reward_explanation", ""), 
        reason_points=info.get("reason_points", [])
    )
    return StepResult(
        observation=obs, 
        reward=reward, 
        done=done, 
        info=info, 
        before_state=info.get("before_state", {}), 
        after_state=info.get("after_state", {})
    )

@app.get("/state", response_model=Observation)
def get_state():
    return env.get_observation()

@app.post("/suggest")
def suggest(mode: str = "llm"):
    obs = env.get_observation()
    # Agent expects Observation dict for LLM processing
    return agent.get_action(obs.model_dump(), mode=mode)

@app.post("/scenario")
def set_scenario(scenario: Scenario):
    env.set_scenario(scenario)
    return {"status": "success", "scenario": scenario}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
