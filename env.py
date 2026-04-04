"""
Cloud Job Scheduler Environment
Main environment implementation with proper OpenEnv API compliance.
"""

import copy
from typing import Dict, Tuple, Any, Optional, List
from models import (
    Observation, Action, Reward, ClusterState, Job, Node, JobPriority, EpisodeResult
)
from tasks import init_static_batch, init_priority_scheduling, init_dynamic_dependencies


class CloudJobSchedulerEnv:
    """
    Dynamic Cloud Job Scheduler Environment.
    
    Simulates a cluster management system where the agent schedules jobs to nodes
    while respecting resource constraints, priorities, deadlines, and dependencies.
    """
    
    def __init__(self, task_id: str = "schedule_static_batch") -> None:
        """
        Initialize the environment.
        
        Args:
            task_id: One of "schedule_static_batch", "schedule_with_priorities", 
                     or "schedule_with_dependencies"
        """
        self.task_id: str = task_id
        self.task_config: Dict[str, Any] = self._load_task_config(task_id)
        
        self.nodes: List[Node] = []
        self.pending_queue: List[Job] = []
        self.running_jobs: Dict[str, Dict[str, Any]] = {}
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        self.current_time: int = 0
        self.max_steps: int = self.task_config["max_steps"]
        self.step_count: int = 0
        self.accumulated_rewards: List[float] = []
        self.grader: Any = self.task_config["grader"]
        self.episode_jobs: List[Job] = []
        
    def _load_task_config(self, task_id: str) -> Dict[str, Any]:
        """
        Load task configuration.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task configuration dictionary
        """
        if task_id == "schedule_static_batch":
            return init_static_batch()
        elif task_id == "schedule_with_priorities":
            return init_priority_scheduling()
        elif task_id == "schedule_with_dependencies":
            return init_dynamic_dependencies()
        else:
            raise ValueError(f"Unknown task: {task_id}")
    
    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        
        Returns:
            Observation: Initial cluster state
        """
        self.nodes = copy.deepcopy(self.task_config["nodes"])
        
        all_jobs: List[Job] = copy.deepcopy(self.task_config["jobs"])
        self.episode_jobs = all_jobs
        
        self.pending_queue = [j for j in all_jobs if j.arrival_time == 0]
        
        self.running_jobs = {}
        self.completed_jobs = []
        self.failed_jobs = []
        self.current_time = 0
        self.step_count = 0
        self.accumulated_rewards = []
        
        return self.state()
    
    def state(self) -> Observation:
        """
        Get current observation.
        
        Returns:
            Observation: Current cluster state and valid actions
        """
        cluster_state: ClusterState = ClusterState(
            current_time=self.current_time,
            nodes=self.nodes,
            running_jobs=self.running_jobs,
            completed_jobs=self.completed_jobs,
            failed_jobs=self.failed_jobs,
            pending_queue=self.pending_queue,
        )
        
        valid_actions: List[str] = self._get_valid_actions()
        
        observation: Observation = Observation(
            cluster_state=cluster_state,
            valid_actions=valid_actions,
        )
        
        return observation
    
    def _get_valid_actions(self) -> List[str]:
        """
        Generate list of valid action strings.
        
        Returns:
            List of valid action strings
        """
        valid: List[str] = []
        valid.append("wait()")
        
        for job in self.pending_queue:
            if self._are_dependencies_satisfied(job):
                for node in self.nodes:
                    if (node.available_cpu >= job.required_cpu and 
                        node.available_ram >= job.required_ram):
                        action_str: str = f"schedule_job(job_id='{job.job_id}',node_id='{node.node_id}')"
                        valid.append(action_str)
        
        return valid
    
    def _are_dependencies_satisfied(self, job: Job) -> bool:
        """
        Check if all dependencies for a job are satisfied (completed).
        
        Args:
            job: Job to check
            
        Returns:
            True if all dependencies completed, False otherwise
        """
        for dep_id in job.dependencies:
            if dep_id not in self.completed_jobs:
                return False
        return True
    
    def step(self, action_str: str) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action_str: String representation of action
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1
        reward_value: float = 0.0
        reward_components: Dict[str, float] = {}
        error_msg: Optional[str] = None
        
        try:
            if action_str == "wait()":
                reward_value, reward_components = self._do_wait()
            elif action_str.startswith("schedule_job"):
                reward_value, reward_components, error_msg = self._parse_and_schedule(action_str)
            else:
                error_msg = f"Unknown action: {action_str}"
                reward_value = -0.5
                reward_components = {"invalid_action_penalty": -0.5}
        except Exception as e:
            error_msg = str(e)
            reward_value = -0.5
            reward_components = {"invalid_action_penalty": -0.5}
        
        self.accumulated_rewards.append(reward_value)
        done: bool = self._check_done()
        
        reward: Reward = Reward(
            value=reward_value,
            components=reward_components,
            info=error_msg or "Action executed successfully"
        )
        
        info: Dict[str, Any] = {
            "step": self.step_count,
            "current_time": self.current_time,
            "error": error_msg,
        }
        
        observation: Observation = self.state()
        return observation, reward, done, info
    
    def _do_wait(self) -> Tuple[float, Dict[str, float]]:
        """
        Execute wait action: advance time by 1 step.
        
        Returns:
            Tuple of (reward_value, reward_components)
        """
        self.current_time += 1
        reward_value: float = 0.0
        reward_components: Dict[str, float] = {}
        
        # Check for job completions
        completed_this_step: List[str] = []
        for job_id in list(self.running_jobs.keys()):
            job_info: Dict[str, Any] = self.running_jobs[job_id]
            if self.current_time >= job_info["end_time"]:
                completed_this_step.append(job_id)
                self.completed_jobs.append(job_id)
                del self.running_jobs[job_id]
                
                job: Optional[Job] = next(
                    (j for j in self.episode_jobs if j.job_id == job_id), None
                )
                if job:
                    base_reward: float = 0.5
                    priority_bonus: float = 0.0
                    if job.priority == JobPriority.HIGH:
                        priority_bonus = 0.2
                    elif job.priority == JobPriority.MEDIUM:
                        priority_bonus = 0.1
                    
                    completion_reward: float = base_reward + priority_bonus
                    reward_value += completion_reward
                    reward_components[f"completion_{job_id}"] = completion_reward
                
                node: Optional[Node] = next(
                    (n for n in self.nodes if n.node_id == job_info["node_id"]), None
                )
                if node and job:
                    node.available_cpu += job.required_cpu
                    node.available_ram += job.required_ram
                    if job_id in node.running_jobs:
                        node.running_jobs.remove(job_id)
        
        # Check for newly arrived jobs
        for job in self.episode_jobs:
            if (job not in self.pending_queue and 
                job.job_id not in self.completed_jobs and 
                job.job_id not in self.failed_jobs and 
                job.job_id not in self.running_jobs and 
                job.arrival_time == self.current_time):
                self.pending_queue.append(job)
        
        # Check for missed deadlines in pending queue
        jobs_to_remove: List[Job] = []
        for job in self.pending_queue:
            if self.current_time > job.deadline:
                jobs_to_remove.append(job)
                self.failed_jobs.append(job.job_id)
                reward_value -= 1.0
                reward_components["deadline_miss"] = reward_components.get("deadline_miss", 0.0) - 1.0
        
        for job in jobs_to_remove:
            self.pending_queue.remove(job)
        
        # Check for missed deadlines in running jobs
        jobs_failed: List[str] = []
        for job_id, job_info in list(self.running_jobs.items()):
            job_obj: Optional[Job] = next(
                (j for j in self.episode_jobs if j.job_id == job_id), None
            )
            if job_obj and self.current_time > job_obj.deadline:
                jobs_failed.append(job_id)
                self.failed_jobs.append(job_id)
                reward_value -= 1.0
                reward_components["deadline_miss"] = reward_components.get("deadline_miss", 0.0) - 1.0
        
        for job_id in jobs_failed:
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
        
        # Small reward if no other reward
        if not reward_components:
            reward_value = 0.05
            reward_components["time_step"] = 0.05
        
        return reward_value, reward_components
    
    def _parse_and_schedule(self, action_str: str) -> Tuple[float, Dict[str, float], Optional[str]]:
        """
        Parse and execute schedule_job action.
        
        Args:
            action_str: Action string to parse
        
        Returns:
            Tuple of (reward_value, reward_components, error_msg)
        """
        try:
            parts: List[str] = action_str.replace("schedule_job(", "").replace(")", "").split(",")
            job_id: str = parts[0].split("=")[1].strip("'\"")
            node_id: str = parts[1].split("=")[1].strip("'\"")
        except Exception as e:
            return 0.0, {"parse_error": -0.3}, f"Failed to parse action: {str(e)}"
        
        job: Optional[Job] = next(
            (j for j in self.pending_queue if j.job_id == job_id), None
        )
        if not job:
            return 0.0, {"invalid_job": -0.3}, f"Job {job_id} not found in pending queue"
        
        node: Optional[Node] = next(
            (n for n in self.nodes if n.node_id == node_id), None
        )
        if not node:
            return 0.0, {"invalid_node": -0.3}, f"Node {node_id} not found"
        
        if not self._are_dependencies_satisfied(job):
            return 0.0, {"unmet_dependencies": -0.5}, f"Job {job_id} has unmet dependencies"
        
        if node.available_cpu < job.required_cpu or node.available_ram < job.required_ram:
            return 0.0, {"insufficient_resources": -0.3}, f"Node {node_id} has insufficient resources"
        
        if self.current_time + job.duration > job.deadline:
            return 0.0, {"infeasible_deadline": -0.2}, f"Job {job_id} cannot complete by deadline"
        
        self.running_jobs[job_id] = {
            "node_id": node_id,
            "start_time": self.current_time,
            "end_time": self.current_time + job.duration,
        }
        
        node.available_cpu -= job.required_cpu
        node.available_ram -= job.required_ram
        node.running_jobs.append(job_id)
        
        self.pending_queue.remove(job)
        
        reward_value: float = 0.1
        reward_components: Dict[str, float] = {"valid_schedule": reward_value}
        
        return reward_value, reward_components, None
    
    def _check_done(self) -> bool:
        """
        Check if episode is done.
        
        Returns:
            True if episode is done, False otherwise
        """
        if self.step_count >= self.max_steps:
            return True
        
        total_jobs: int = len(self.episode_jobs)
        resolved: int = len(self.completed_jobs) + len(self.failed_jobs)
        
        if resolved == total_jobs and len(self.pending_queue) == 0 and len(self.running_jobs) == 0:
            return True
        
        return False
    
    def _compute_final_metrics(self) -> Dict[str, float]:
        """
        Compute final episode metrics.
        
        Returns:
            Dictionary of final metrics
        """
        total_jobs: int = len(self.episode_jobs)
        completed: int = len(self.completed_jobs)
        failed: int = len(self.failed_jobs)
        
        sla_compliance: float = completed / total_jobs if total_jobs > 0 else 0.0
        resource_efficiency: float = min(1.0, sum(self.accumulated_rewards) / max(1.0, total_jobs))
        makespan: int = self.current_time
        
        return {
            "sla_compliance": sla_compliance,
            "resource_efficiency": max(0.0, resource_efficiency),
            "makespan": makespan,
            "completed_count": completed,
            "failed_count": failed,
        }
    
    def get_final_result(self) -> EpisodeResult:
        """
        Get final episode result for grading.
        
        Returns:
            EpisodeResult with all metrics and final score
        """
        metrics: Dict[str, float] = self._compute_final_metrics()
        
        temp_result: EpisodeResult = EpisodeResult(
            success=len(self.failed_jobs) == 0,
            steps=self.step_count,
            score=0.0,
            rewards=self.accumulated_rewards,
            sla_compliance=metrics["sla_compliance"],
            resource_efficiency=metrics["resource_efficiency"],
            completed_count=metrics["completed_count"],
            failed_count=metrics["failed_count"],
            makespan=metrics["makespan"],
        )
        
        final_state: ClusterState = ClusterState(
            current_time=self.current_time,
            nodes=self.nodes,
            running_jobs=self.running_jobs,
            completed_jobs=self.completed_jobs,
            failed_jobs=self.failed_jobs,
            pending_queue=self.pending_queue,
        )
        
        score: float = self.grader.grade(final_state, temp_result)
        
        return EpisodeResult(
            success=len(self.failed_jobs) == 0,
            steps=self.step_count,
            score=score,
            rewards=self.accumulated_rewards,
            sla_compliance=metrics["sla_compliance"],
            resource_efficiency=metrics["resource_efficiency"],
            completed_count=metrics["completed_count"],
            failed_count=metrics["failed_count"],
            makespan=metrics["makespan"],
        )