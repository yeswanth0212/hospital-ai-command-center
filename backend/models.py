from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class Scenario(str, Enum):
    NORMAL = "NORMAL"
    OVERLOAD = "OVERLOAD"
    SHORTAGE = "SHORTAGE"
    STABLE = "STABLE"

class ActionType(str, Enum):
    ICU = "ICU"
    GENERAL = "GENERAL"
    WAIT = "WAIT"
    TRANSFER = "TRANSFER"

class PatientStatus(str, Enum):
    WAITING = "WAITING"
    ALLOCATED_GENERAL = "ALLOCATED_GENERAL"
    ALLOCATED_ICU = "ALLOCATED_ICU"
    CRITICAL = "CRITICAL"
    DECEASED = "DECEASED"
    RECOVERED = "RECOVERED"

class Patient(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str = "Unknown"
    severity: int = Field(ge=0)
    waiting_time: int = 0
    condition: str = "general"
    status: PatientStatus = PatientStatus.WAITING
    deterioration_rate: float = 0.1

class Action(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    patient_id: Optional[str] = None
    action_type: ActionType
    target_bed_type: Optional[str] = None

class Observation(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    icu_available: int = 0
    general_available: int = 0
    patients: List[Patient] = []
    current_step: int = 0
    cumulative_reward: float = 0.0
    deaths: int = 0
    survival_rate: float = 0.0
    resource_util: float = 0.0
    fairness_index: float = 1.0
    efficiency_score: float = 0.0
    alerts: List[str] = []
    scenario: Scenario = Scenario.NORMAL

class Reward(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    value: float
    explanation: str
    reason_points: List[str] = []

class StepResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}
    before_state: Dict[str, Any] = {}
    after_state: Dict[str, Any] = {}
    confidence: float = 1.0
