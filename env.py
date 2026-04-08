
from pydantic import BaseModel
from typing import List
import random

class Observation(BaseModel):
    task_id: str
    complexity: float
    urgency: float
    ai_reliability_est: float
    remaining_budget: float
    remaining_steps: int
    current_risk_level: float
    recent_history: List[str]

class Action(BaseModel):
    strategy: str

class StrategicRoutingEnv:
    def __init__(self, budget=50.0, max_steps=10):
        self.initial_budget = budget
        self.max_steps = max_steps
        self.tasks_database = [
            {'name': 'Medical Imaging', 'base_comp': 0.8, 'base_urg': 0.4, 'cost_h': 15.0},
            {'name': 'Spam Triage', 'base_comp': 0.2, 'base_urg': 0.9, 'cost_h': 2.0},
            {'name': 'Contract Analysis', 'base_comp': 0.6, 'base_urg': 0.3, 'cost_h': 8.0}
        ]
        self.reset()

    def reset(self):
        self.budget, self.current_step, self.risk_level = self.initial_budget, 0, 0.0
        self.history = []
        self.episode_tasks = random.choices(self.tasks_database, k=self.max_steps)
        return self._get_obs()

    def _get_obs(self):
        if self.current_step >= self.max_steps:
            return Observation(task_id='done', complexity=0, urgency=0, ai_reliability_est=0,
                               remaining_budget=self.budget, remaining_steps=0, current_risk_level=self.risk_level, recent_history=self.history[-3:])
        t = self.episode_tasks[self.current_step]
        return Observation(task_id=t['name'], complexity=t['base_comp'], urgency=t['base_urg'],
                           ai_reliability_est=max(0.1, 1.0 - t['base_comp']), remaining_budget=self.budget,
                           remaining_steps=self.max_steps - self.current_step, current_risk_level=self.risk_level, recent_history=self.history[-3:])

    def add_external_task(self, name: str, complexity: float, description: str):
        self.tasks_database.append({
            'name': name,
            'base_comp': float(complexity),
            'base_urg': 0.5,
            'cost_h': 10.0
        })

    def step(self, action: Action):
        if self.current_step >= self.max_steps:
             return self._get_obs(), 0, True, {}
        
        task = self.episode_tasks[self.current_step]
        cost = 10.0 if action.strategy == 'human_expert' else 1.0
        success = random.random() > (task['base_comp'] if action.strategy != 'human_expert' else 0.05)
        reward = 1.0 + task['base_comp'] if success else -2.0
        
        self.budget -= cost
        self.risk_level += 0.2 if not success else 0.0
        self.history.append(f"{task['name']}:{action.strategy}")
        self.current_step += 1
        
        done = self.current_step >= self.max_steps or self.budget <= 0
        return self._get_obs(), reward, done, {}
