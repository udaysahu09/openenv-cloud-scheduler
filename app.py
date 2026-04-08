#!/usr/bin/env python3
import os
import json
import time
import datetime
from env import CloudJobSchedulerEnv
import gradio as gr
from fastapi import FastAPI
import uvicorn

# Global environment
current_env = None

# FastAPI app
app = FastAPI()

# Reset function
@app.post('/openenv/reset')
async def reset_endpoint():
    """Reset environment endpoint"""
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
        return {"status": "error", "message": str(e)}

# Step function
@app.post('/openenv/step')
async def step_endpoint(request_data: dict):
    """Step environment"""
    global current_env
    try:
        action = request_data.get('action', 'wait()')
        obs, reward, done, info = current_env.step(action)
        return {
            "status": "success",
            "reward": float(reward.value) if hasattr(reward, 'value') else reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Health check
@app.get('/health')
async def health():
    return {"status": "ok"}

# Gradio UI function
def run_cloud_job(job_name, job_type, priority, progress=gr.Progress()):
    if not job_name:
        return "Error: Please provide a Job Name."

    progress(0.1, desc="Initializing cloud environment...")
    time.sleep(1)
    
    progress(0.3, desc=f"Allocating resources for {job_type}...")
    time.sleep(1.5)
    
    progress(0.5, desc=f"Executing {job_name}...")
    execution_time = 2 if priority == "High (Fast)" else 4
    for i in range(execution_time):
        time.sleep(1)
        progress(0.5 + (0.4 * (i / execution_time)), desc=f"Processing step {i+1} of {execution_time}...")
        
    progress(1.0, desc="Cleaning up and saving logs...")
    time.sleep(1)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = f"✅ [{timestamp}] SUCCESS: Job completed!\n"
    log += f"➜ Job Name: {job_name}\n"
    log += f"➜ Task Type: {job_type}\n"
    log += f"➜ Priority Level: {priority}\n"
    log += "➜ Status: All tasks executed successfully on Openenv Cloud."
    
    return log

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# ☁️ Openenv Cloud Job Scheduler")
    gr.Markdown("Submit a task to be executed on the cloud environment.")
    
    with gr.Row():
        with gr.Column():
            job_name = gr.Textbox(label="Job Name", placeholder="e.g., nightly-database-backup")
            job_type = gr.Dropdown(
                choices=["Data Backup", "AI Model Training", "Web Scraping", "System Update"], 
                label="Task Type", 
                value="Data Backup"
            )
            priority = gr.Radio(
                choices=["Standard", "High (Fast)"], 
                label="Execution Priority", 
                value="Standard"
            )
            submit_btn = gr.Button("🚀 Schedule & Run Job", variant="primary")
            
        with gr.Column():
            output_log = gr.Textbox(label="Execution Logs", lines=10)
            
    submit_btn.click(
        fn=run_cloud_job, 
        inputs=[job_name, job_type, priority], 
        outputs=output_log
    )

# Mount Gradio on FastAPI
gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=7860)