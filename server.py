from fastapi import FastAPI
from env import StrategicRoutingEnv, Action

app = FastAPI(root_path="/")
env = StrategicRoutingEnv()

# ✅ ADD THIS
@app.get("/")
def home():
    return {"message": "Human-AI Optimizer API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post('/reset')
def reset():
    return env.reset()

@app.post('/add_task')
def add_task(name: str, diff: float, desc: str):
    env.add_external_task(name, diff, desc)
    return {'status': 'added'}

@app.post('/step')
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        'observation': obs,
        'reward': reward,
        'done': done
    }