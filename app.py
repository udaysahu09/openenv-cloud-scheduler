import gradio as gr
import time
import datetime
from flask import Flask, jsonify, request
from threading import Thread
from env import CloudJobSchedulerEnv
import os

# Initialize Flask app
app = Flask(__name__)

# Global environment instance
current_env = None

# Flask Routes (for grader)
@app.route('/openenv/reset', methods=['POST'])
def reset():
    """Reset the job scheduling environment"""
    global current_env
    try:
        task_id = os.getenv("TASK_ID", "schedule_static_batch")
        current_env = CloudJobSchedulerEnv(task_id=task_id)
        obs = current_env.reset()
        
        return jsonify({
            "status": "success",
            "message": "Environment reset successfully",
            "observation": str(obs)
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/openenv/step', methods=['POST'])
def step():
    """Execute one step in the environment"""
    global current_env
    try:
        action = request.json.get('action', 'wait()')
        obs, reward, done, info = current_env.step(action)
        
        return jsonify({
            "status": "success",
            "reward": reward.value,
            "done": done,
            "info": info
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"}), 200

# Gradio UI Functions
def run_cloud_job(job_name, job_type, priority, progress=gr.Progress()):
    if not job_name:
        return "Error: Please provide a Job Name."

    # Step 1: Initialization
    progress(0.1, desc="Initializing cloud environment...")
    time.sleep(1)
    
    # Step 2: Resource Allocation
    progress(0.3, desc=f"Allocating resources for {job_type}...")
    time.sleep(1.5)
    
    # Step 3: Execution (takes longer if priority is low)
    progress(0.5, desc=f"Executing {job_name}...")
    execution_time = 2 if priority == "High (Fast)" else 4
    for i in range(execution_time):
        time.sleep(1)
        progress(0.5 + (0.4 * (i / execution_time)), desc=f"Processing step {i+1} of {execution_time}...")
        
    # Step 4: Cleanup
    progress(1.0, desc="Cleaning up and saving logs...")
    time.sleep(1)
    
    # Generate the final log output
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = f"✅ [{timestamp}] SUCCESS: Job completed!\n"
    log += f"➜ Job Name: {job_name}\n"
    log += f"➜ Task Type: {job_type}\n"
    log += f"➜ Priority Level: {priority}\n"
    log += "➜ Status: All tasks executed successfully on Openenv Cloud."
    
    return log

# Create Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ Openenv Cloud Job Scheduler")
    gr.Markdown("Submit a task to be executed on the cloud environment. Monitor the progress in real-time.")
    
    with gr.Row():
        # Left column: User Inputs
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
            
        # Right column: Output Logs
        with gr.Column():
            output_log = gr.Textbox(label="Execution Logs", lines=10)
            
    # Connect the button to the Python function
    submit_btn.click(
        fn=run_cloud_job, 
        inputs=[job_name, job_type, priority], 
        outputs=output_log
    )

# Mount Gradio app on Flask
gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860, debug=False)