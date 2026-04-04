#!/usr/bin/env python3
"""
Baseline inference script for Cloud Job Scheduler.
Uses an LLM to make scheduling decisions.
"""

import os
import sys
import json
import re
from typing import Optional, List
from env import CloudJobSchedulerEnv
from models import Observation

try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = None
    AzureOpenAI = None


class SchedulerAgent:
    """Agent that uses LLM for scheduling decisions."""
    
    def __init__(self, model_name: str = "gpt-4", api_base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            model_name: Model name
            api_base_url: API base URL
            api_key: API key for authentication
        """
        self.model_name = model_name
        
        # Determine which API to use
        if api_base_url and "openai.azure.com" in api_base_url:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=api_base_url,
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
            )
    
    def _format_observation(self, obs: Observation) -> str:
        """Format observation into a readable prompt."""
        state = obs.cluster_state
        
        # Format nodes
        nodes_str = "Cluster Nodes:\n"
        for node in state.nodes:
            nodes_str += f"  {node.node_id}: {node.available_cpu}/{node.total_cpu} CPU, {node.available_ram}/{node.total_ram} GB RAM, running {len(node.running_jobs)} jobs\n"
        
        # Format running jobs
        running_str = "Running Jobs:\n"
        if state.running_jobs:
            for job_id, info in state.running_jobs.items():
                running_str += f"  {job_id} on {info['node_id']} (ends at t={info['end_time']})\n"
        else:
            running_str += "  (none)\n"
        
        # Format pending queue
        queue_str = "Pending Queue:\n"
        if state.pending_queue:
            for job in state.pending_queue:
                deps_str = f", depends on {job.dependencies}" if job.dependencies else ""
                queue_str += f"  {job.job_id}: {job.required_cpu}CPU/{job.required_ram}GB, priority={job.priority}, deadline=t{job.deadline}{deps_str}\n"
        else:
            queue_str += "  (none)\n"
        
        # Format valid actions
        actions_str = "Valid Actions:\n"
        for action in obs.valid_actions[:10]:  # Show first 10 valid actions
            actions_str += f"  - {action}\n"
        if len(obs.valid_actions) > 10:
            actions_str += f"  ... and {len(obs.valid_actions) - 10} more\n"
        
        prompt = f"""Current Cluster State (Time={state.current_time}):

{nodes_str}
{running_str}
{queue_str}
{actions_str}

Completed Jobs: {len(state.completed_jobs)}
Failed Jobs: {len(state.failed_jobs)}

Choose the next action to optimize job scheduling. Prioritize:
1. High-priority jobs
2. Jobs with approaching deadlines
3. Respect dependencies and resource constraints

Respond with ONLY the action command, nothing else. Example: schedule_job(job_id='job_0',node_id='node_0') or wait()
"""
        return prompt
    
    def decide(self, obs: Observation) -> str:
        """
        Get LLM decision for next action.
        
        Args:
            obs: Current observation
        
        Returns:
            Action string
        """
        if not self.client:
            # Fallback: just wait
            return "wait()"
        
        prompt = self._format_observation(obs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a cluster scheduling expert. Make optimal scheduling decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100,
            )
            
            action = response.choices[0].message.content.strip()
            
            # Clean up response
            if "schedule_job" in action or "wait()" in action:
                # Extract the action
                if "schedule_job" in action:
                    match = re.search(r"schedule_job\(.*?\)", action)
                    if match:
                        action = match.group(0)
                    else:
                        action = "wait()"
                elif "wait" in action:
                    action = "wait()"
            else:
                action = "wait()"
            
            return action
        except Exception as e:
            print(f"Error calling LLM: {e}", file=sys.stderr)
            return "wait()"


def main():
    """Main inference loop."""
    # Get configuration from environment
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4")
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
    
    # Default task
    task_id = os.getenv("TASK_ID", "schedule_static_batch")
    
    # Initialize environment and agent
    env = CloudJobSchedulerEnv(task_id=task_id)
    agent = SchedulerAgent(model_name=model_name, api_base_url=api_base_url, api_key=api_key)
    
    # Print START
    print(f"[START] task={task_id} env=cloud_job_scheduler model={model_name}")
    
    obs = env.reset()
    done = False
    step = 0
    all_rewards = []
    
    while not done and step < env.max_steps:
        step += 1
        
        # Get action from agent
        action = agent.decide(obs)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        error_msg = info.get("error")
        error_str = f'"{error_msg}"' if error_msg else "null"
        
        all_rewards.append(reward.value)
        
        # Print STEP
        print(f"[STEP] step={step} action={action} reward={reward.value:.2f} done={str(done).lower()} error={error_str}")
    
    # Get final result
    result = env.get_final_result()
    
    # Print END
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    print(f"[END] success={str(result.success).lower()} steps={result.steps} score={result.score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    main()