import gradio as gr
import numpy as np
from stable_baselines3 import PPO
import os

# Define action mapping
action_mapping = ['ai_fast', 'ai_balanced', 'human_expert', 'defer']

# Load the trained model
# Ensure you upload 'ppo_strategic_orchestrator.zip' to Hugging Face as well
model = PPO.load("ppo_strategic_orchestrator")

def final_predict_ui(name, comp, urg):
    ai_rel = max(0.1, 1.0 - comp)
    # [complexity, urgency, ai_reliability_est, remaining_budget, remaining_steps, current_risk_level]
    obs = np.array([comp, urg, ai_rel, 50.0, 1, 0.0], dtype=np.float32)

    action_idx, _ = model.predict(obs, deterministic=True)
    strategy = action_mapping[action_idx]

    display_names = {
        'human_expert': '🧑‍⚕️ Human Expert',
        'ai_fast': '⚡ AI Fast',
        'ai_balanced': '⚖️ AI Balanced',
        'defer': "⏳ Defer"
    }
    return f"The RL Agent says: {display_names.get(strategy, strategy)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🎓 RL Task Router Deployment")
    name = gr.Textbox(label="Task Name")
    comp = gr.Slider(0, 1, value=0.5, label="Complexity")
    urg = gr.Slider(0, 1, value=0.5, label="Urgency")
    predict_btn = gr.Button("Get Routing Decision")
    output_text = gr.Label(label="Decision")
    predict_btn.click(final_predict_ui, inputs=[name, comp, urg], outputs=output_text)

demo.launch()