# app.py
from flask import Flask, jsonify, request
from env import CloudJobSchedulerEnv
import os

app = Flask(__name__)

# Global environment instance
current_env = None

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
