import os
import requests
from openai import OpenAI

# ================= CONFIG =================
API_BASE_URL = os.getenv("API_BASE_URL", "https://human-ai-optimizer.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")

TASK_NAME = "human_vs_ai_decision"
BENCHMARK = "custom_env"
MAX_STEPS = 10

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=API_KEY)

# ================= LOGGING =================
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ================= MODEL =================
def get_decision(task, description):
    prompt = f"""
    Task: {task}
    Description: {description}

    Should this be done by AI or Human?
    Reply ONLY with 'ai' or 'human'.
    """

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return res.choices[0].message.content.strip().lower()
    except:
        return "ai"

# ================= MAIN =================
def main():
    log_start()

    rewards = []
    steps_taken = 0
    success = False

    # RESET ENV
    state = requests.post(f"{API_BASE_URL}/reset").json()

    for step in range(1, MAX_STEPS + 1):
        task = state["task_id"]
        description = state.get("description", "")

        decision = get_decision(task, description)

        result = requests.post(
            f"{API_BASE_URL}/step",
            json={"strategy": decision}
        ).json()

        reward = result["reward"]
        done = result["done"]

        rewards.append(reward)
        steps_taken = step

        log_step(step, decision, reward, done, None)

        if done:
            break

        state = result["observation"]

    success = sum(rewards) > 2  # simple threshold

    log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    main()