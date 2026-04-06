# Smart Hospital AI Command Center

A high-fidelity simulation and evaluation platform designed to test the limits of AI-driven clinical allocation. Built specifically for the **OpenEnv Competition**, this project models the intense environment of a hospital emergency triage queue, demanding that autonomous agents efficiently allocate scarce ICU and General Ward beds to patients suffering from varying degrees of life-threatening conditions.

## Problem Statement & Motivation

During mass casualty events or severe hospital shortages, triage nurses must make split-second, life-or-death resource allocations. This OpenEnv simulation tests if Large Language Models or Reinforcement Learning agents can master these stakes. Agents are challenged to keep patients alive, minimize bed wastage, and demonstrate holistic resource awareness across simulated timesteps.

## OpenEnv Specifics

### Action Space (Discrete)
- **ICU**: Allocate an Intensive Care Unit bed to a specific patient.
- **GENERAL**: Allocate a General Ward bed to a specific patient.
- **WAIT**: Do nothing and wait out the timestep. (Patients left waiting will slowly deteriorate; severe patients run the risk of expiring).

### Observation Space
The environment emits a continuous state dict capturing:
- `icu_available`: Int (Beds remaining out of 5)
- `general_available`: Int (Beds remaining out of 15)
- `patients`: List of queued Patients (tracking severity out of 10, condition type, and timeline)
- `deaths` / `survival_rate` / `efficiency_score`: System metrics

### Competition Tasks
The agent's logic is rigorously tested against three grader tasks:
1. **Easy (Prioritize High Severity)**: Focuses purely on the agent's ability to identify patients with `severity >= 7` and allocate them to the ICU.
2. **Medium (Efficient Resource Allocation)**: Penalizes "safe" over-allocations (like wasting ICU beds on non-critical patients) and rewards precise bed matching. 
3. **Hard (Minimize Deaths & Maximize Throughput)**: A holistic throughput test heavily penalizing patient deaths while aggressively expecting efficient bed turnover and high survival rates.

## Installation & Setup

Ensure you have Python 3.10+ installed.

```bash
# Install dependencies
pip install fastapi uvicorn pydantic requests openai
```

## Running the Architecture

### 1. Start the Environment API & Dashboard
Run the FastAPI backend which mounts the simulation state:
```bash
python backend/main.py
```
> *The interactive, glass-morphism dashboard can be accessed locally to visually inspect your agent's behavior.*

### 2. Run the Baseline Inference Agent
The `inference.py` script serves as the baseline submission. It triggers episode loops, pushes actions to the API, and evaluates the final trajectory against the OpenEnv grader logic.

Set your environment variables, then run the baseline:
```bash
# Windows
set API_BASE_URL=http://localhost:8000
set OPENAI_API_KEY=your_key_here
python inference.py

# Linux/Mac
export API_BASE_URL="http://localhost:8000"
export OPENAI_API_KEY="your_key_here"
python inference.py
```

### Example Automated Output

```text
[START] Simulation Episode
[STEP] Action: ICU, Patient: e5f37491
[STEP] Action: WAIT, Patient: None
[END] Episode Complete
--------------------------------------------------
EVALUATION RESULTS
--------------------------------------------------
Task 1 (Easy) Score:   0.88 / 1.00
Task 2 (Medium) Score: 0.74 / 1.00
Task 3 (Hard) Score:   0.61 / 1.00
```

## Docker / Hugging Face Spaces Deployment

The simulator comes completely Dockerized with standard exposure mapped to `:7860`, allowing native deployments right into Hugging Face Spaces without adjustments.

```bash
docker build -t smart-hospital .
docker run -p 7860:7860 smart-hospital
```
