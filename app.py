#!/usr/bin/env python3
import os
import json
from env import CloudJobSchedulerEnv
from models import Observation

# Global environment
current_env = None

def reset():
    """Reset environment"""
    global current_env
    try:
        task_id = os.getenv("TASK_ID", "schedule_static_batch")
        current_env = CloudJobSchedulerEnv(task_id=task_id)
        obs = current_env.reset()
        return {"status": "success", "message": "Environment reset"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def step(action):
    """Step environment"""
    global current_env
    try:
        obs, reward, done, info = current_env.step(action)
        return {
            "status": "success",
            "reward": float(reward.value),
            "done": done,
            "info": info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print(json.dumps(reset()))