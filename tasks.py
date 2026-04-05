from typing import Dict, Any, List
from models import ClusterState, Job, Node, JobPriority, EpisodeResult


class TaskGrader:
    def grade(self, final_state: ClusterState, episode_result: EpisodeResult) -> float:
        raise NotImplementedError


class StaticBatchGrader(TaskGrader):
    def grade(self, final_state: ClusterState, episode_result: EpisodeResult) -> float:
        total_jobs = episode_result.completed_count + episode_result.failed_count
        if total_jobs == 0:
            return 0.0
        completion_rate = episode_result.completed_count / total_jobs
        score = completion_rate
        if episode_result.failed_count == 0:
            score = min(1.0, score + 0.2)
        efficiency_penalty = (1.0 - episode_result.resource_efficiency) * 0.1
        score -= efficiency_penalty
        return max(0.0, min(1.0, score))


class PrioritySchedulingGrader(TaskGrader):
    def grade(self, final_state: ClusterState, episode_result: EpisodeResult) -> float:
        sla_score = episode_result.sla_compliance * 0.7
        efficiency_score = episode_result.resource_efficiency * 0.3
        total_score = sla_score + efficiency_score
        return max(0.0, min(1.0, total_score))


class DynamicDependenciesGrader(TaskGrader):
    def grade(self, final_state: ClusterState, episode_result: EpisodeResult) -> float:
        sla_score = episode_result.sla_compliance * 0.6
        efficiency_score = episode_result.resource_efficiency * 0.3
        dependency_score = min(episode_result.sla_compliance, 1.0) * 0.1
        total_score = sla_score + efficiency_score + dependency_score
        return max(0.0, min(1.0, total_score))


def init_static_batch() -> Dict[str, Any]:
    nodes = [Node(node_id=f"node_{i}", total_cpu=4.0, total_ram=16.0, available_cpu=4.0, available_ram=16.0, running_jobs=[]) for i in range(4)]
    jobs = [
        Job(job_id="job_0", required_cpu=1.0, required_ram=2.0, duration=5, priority=JobPriority.HIGH, deadline=20, dependencies=[]),
        Job(job_id="job_1", required_cpu=2.0, required_ram=4.0, duration=8, priority=JobPriority.MEDIUM, deadline=25, dependencies=[]),
        Job(job_id="job_2", required_cpu=1.5, required_ram=3.0, duration=6, priority=JobPriority.LOW, deadline=22, dependencies=[]),
        Job(job_id="job_3", required_cpu=2.0, required_ram=6.0, duration=10, priority=JobPriority.HIGH, deadline=28, dependencies=[]),
        Job(job_id="job_4", required_cpu=1.0, required_ram=2.0, duration=4, priority=JobPriority.MEDIUM, deadline=18, dependencies=[]),
        Job(job_id="job_5", required_cpu=3.0, required_ram=8.0, duration=12, priority=JobPriority.LOW, deadline=30, dependencies=[]),
        Job(job_id="job_6", required_cpu=1.5, required_ram=4.0, duration=7, priority=JobPriority.HIGH, deadline=24, dependencies=[]),
        Job(job_id="job_7", required_cpu=2.0, required_ram=5.0, duration=9, priority=JobPriority.MEDIUM, deadline=26, dependencies=[]),
        Job(job_id="job_8", required_cpu=1.0, required_ram=3.0, duration=5, priority=JobPriority.LOW, deadline=20, dependencies=[]),
        Job(job_id="job_9", required_cpu=2.5, required_ram=7.0, duration=11, priority=JobPriority.HIGH, deadline=27, dependencies=[]),
    ]
    return {"nodes": nodes, "jobs": jobs, "grader": StaticBatchGrader(), "max_steps": 100}


def init_priority_scheduling() -> Dict[str, Any]:
    nodes = [Node(node_id=f"node_{i}", total_cpu=6.0, total_ram=24.0, available_cpu=6.0, available_ram=24.0, running_jobs=[]) for i in range(3)]
    jobs = []
    job_configs = [(1.0, 2.0, 4, JobPriority.HIGH, 12), (2.0, 4.0, 6, JobPriority.HIGH, 15), (1.5, 3.0, 5, JobPriority.MEDIUM, 20), (3.0, 8.0, 10, JobPriority.LOW, 30), (2.0, 5.0, 7, JobPriority.MEDIUM, 18), (1.0, 2.0, 3, JobPriority.HIGH, 10), (2.5, 6.0, 8, JobPriority.LOW, 25), (1.5, 4.0, 6, JobPriority.MEDIUM, 19), (3.5, 9.0, 12, JobPriority.LOW, 35), (1.0, 3.0, 4, JobPriority.HIGH, 13), (2.0, 5.0, 7, JobPriority.MEDIUM, 21), (2.5, 6.0, 8, JobPriority.HIGH, 16)]
    for i, (cpu, ram, duration, priority, deadline) in enumerate(job_configs):
        jobs.append(Job(job_id=f"job_{i}", required_cpu=cpu, required_ram=ram, duration=duration, priority=priority, deadline=deadline, dependencies=[]))
    return {"nodes": nodes, "jobs": jobs, "grader": PrioritySchedulingGrader(), "max_steps": 120}


def init_dynamic_dependencies() -> Dict[str, Any]:
    nodes = [Node(node_id=f"node_{i}", total_cpu=8.0, total_ram=32.0, available_cpu=8.0, available_ram=32.0, running_jobs=[]) for i in range(3)]
    jobs = [
        Job(job_id="job_0", required_cpu=2.0, required_ram=4.0, duration=5, priority=JobPriority.HIGH, deadline=15, dependencies=[], arrival_time=0),
        Job(job_id="job_1", required_cpu=2.0, required_ram=4.0, duration=6, priority=JobPriority.MEDIUM, deadline=20, dependencies=["job_0"], arrival_time=0),
        Job(job_id="job_2", required_cpu=1.5, required_ram=3.0, duration=4, priority=JobPriority.LOW, deadline=25, dependencies=["job_1"], arrival_time=0),
        Job(job_id="job_3", required_cpu=2.0, required_ram=5.0, duration=7, priority=JobPriority.HIGH, deadline=20, dependencies=[], arrival_time=5),
        Job(job_id="job_4", required_cpu=1.5, required_ram=3.0, duration=5, priority=JobPriority.MEDIUM, deadline=22, dependencies=["job_3"], arrival_time=5),
        Job(job_id="job_5", required_cpu=2.5, required_ram=6.0, duration=8, priority=JobPriority.LOW, deadline=30, dependencies=[], arrival_time=10),
        Job(job_id="job_6", required_cpu=2.0, required_ram=4.0, duration=6, priority=JobPriority.HIGH, deadline=25, dependencies=["job_5"], arrival_time=10),
        Job(job_id="job_7", required_cpu=1.0, required_ram=2.0, duration=3, priority=JobPriority.MEDIUM, deadline=28, dependencies=["job_6"], arrival_time=10),
    ]
    return {"nodes": nodes, "jobs": jobs, "grader": DynamicDependenciesGrader(), "max_steps": 150}


def grade_static_batch(final_state: ClusterState, episode_result: EpisodeResult) -> float:
    return StaticBatchGrader().grade(final_state, episode_result)


def grade_priority_scheduling(final_state: ClusterState, episode_result: EpisodeResult) -> float:
    return PrioritySchedulingGrader().grade(final_state, episode_result)


def grade_dynamic_dependencies(final_state: ClusterState, episode_result: EpisodeResult) -> float:
    return DynamicDependenciesGrader().grade(final_state, episode_result)
