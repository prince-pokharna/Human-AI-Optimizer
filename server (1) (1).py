
from fastapi import FastAPI
from env import TaskRoutingEnv, Action

app = FastAPI()
env = TaskRoutingEnv()

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }
