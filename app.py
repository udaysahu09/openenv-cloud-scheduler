#!/usr/bin/env python3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from env import CloudJobSchedulerEnv

app = FastAPI()

# Global environment instance
current_env = None

@app.post('/openenv/reset')
async def reset():
    """Reset the job scheduling environment - POST endpoint"""
    global current_env
    try:
        task_id = os.getenv("TASK_ID", "schedule_static_batch")
        current_env = CloudJobSchedulerEnv(task_id=task_id)
        obs = current_env.reset()
        
        return {
            "status": "success",
            "message": "Environment reset successfully",
            "observation": str(obs)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post('/openenv/step')
async def step(request: Request):
    """Execute one step in the environment"""
    global current_env
    try:
        data = await request.json()
        action = data.get('action', 'wait()')
        obs, reward, done, info = current_env.step(action)
        
        return {
            "status": "success",
            "reward": reward.value if hasattr(reward, 'value') else reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get('/health')
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get('/')
async def root():
    """Root endpoint"""
    return {"message": "OpenEnv Cloud Scheduler API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)