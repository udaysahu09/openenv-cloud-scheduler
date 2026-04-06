# Dynamic Cloud Job Scheduler - OpenEnv Benchmark

A realistic, production-grade environment for evaluating AI agents on cloud resource management and job scheduling problems.

## 🎯 Environment Overview

### Motivation

Job scheduling in distributed computing systems (Kubernetes, SLURM, cloud data centers) is a complex optimization problem that requires:
- **Resource allocation** under hard constraints (CPU, memory)
- **Priority-aware decisions** (SLA levels, business criticality)
- **Deadline enforcement** (time-sensitive workloads)
- **Dependency management** (data pipelines, workflows)

This environment provides a **realistic simulation** where AI agents act as cluster managers, making sequential scheduling decisions that directly impact system efficiency and SLA compliance.

### Real-World Relevance

- **Kubernetes cluster auto-scaling**: Similar to kube-scheduler decisions
- **HPC job scheduling**: Analogous to SLURM resource allocation
- **Serverless platforms**: Job queueing and node selection
- **Data center management**: Cost optimization under resource constraints

---

## 📊 Observation Space

### Cluster State
```python
{
  "current_time": int,           # Current simulation timestep
  "nodes": [                     # List of compute nodes
    {
      "node_id": str,
      "total_cpu": float,        # Total CPU cores (e.g., 4.0)
      "total_ram": float,        # Total RAM in GB (e.g., 16.0)
      "available_cpu": float,    # Currently available CPU
      "available_ram": float,    # Currently available RAM
      "running_jobs": [str]      # Job IDs currently running
    }
  ],
  "running_jobs": {              # Currently executing jobs
    "job_id": {
      "node_id": str,
      "start_time": int,
      "end_time": int
    }
  },
  "completed_jobs": [str],       # Successfully finished jobs
  "failed_jobs": [str],          # Jobs that missed deadline
  "pending_queue": [             # Jobs awaiting scheduling
    {
      "job_id": str,
      "required_cpu": float,
      "required_ram": float,
      "duration": int,           # Time steps to complete
      "priority": "high" | "elevated" | "medium" | "low",
      "deadline": int,           # Absolute deadline (time step)
      "dependencies": [str],     # Job IDs that must complete first
      "arrival_time": int        # When job enters the system
    }
  ]
}
```

### Valid Actions
The agent receives a list of valid actions in each state:
```python
[
  "schedule_job(job_id='job_0',node_id='node_0')",
  "schedule_job(job_id='job_1',node_id='node_1')",
  "wait()"
]
```

---

## ⚙️ Action Space

### `schedule_job(job_id, node_id)`
Assign a job to a specific node.

**Preconditions:**
- Job is in the pending queue
- All job dependencies are satisfied (completed)
- Node has sufficient available CPU and RAM
- Job can complete before its deadline

**Effects:**
- Job moves from pending queue to running jobs
- Node resources are reserved
- Running job will complete after `duration` timesteps

**Reward:** 
- +0.1 for valid scheduling
- -0.3 to -0.5 for invalid attempts (insufficient resources, unmet dependencies, etc.)

### `wait()`
Advance simulation time by 1 timestep.

**Effects:**
- All running jobs advance by 1 timestep
- Jobs that reach their `end_time` are marked completed
- Freed resources become available on nodes
- New jobs may arrive if `arrival_time == current_time`
- Pending jobs exceeding their `deadline` are marked failed

**Reward:**
- +0.5 per job completion (base) + priority bonus (+0.1 for medium, +0.15 for elevated, +0.2 for high)
- -1.0 per missed deadline (SLA breach)
- +0.05 if no jobs completed (progress reward)

---

## 🎓 Tasks

### Easy Task: `schedule_static_batch`

**Scenario:** A static batch of 10 independent jobs arrives at time 0.

**Difficulty:** No dependencies, no dynamic arrivals, straightforward resource matching.

**Cluster:**
- 4 nodes × (4 CPU, 16 GB RAM each)

**Job Set:**
- 10 independent jobs with varying resource needs
- Deadlines: 12-30 timesteps
- Priorities: mix of low/medium/high

**Success Criteria:**
- Maximize jobs completed by deadline
- Grader returns score proportional to completion rate
- Bonus for 100% SLA compliance

**Expected Solution:**
- Sort by priority and deadline
- Bin-pack jobs onto nodes
- Schedule high-priority jobs earlier

**Baseline Performance:** 0.85+ (with good heuristics)

---

### Medium Task: `schedule_with_priorities`

**Scenario:** 12 jobs with mixed priorities that require intelligent prioritization.

**Difficulty:** Must balance high-priority jobs (tighter deadlines) with resource constraints.

**Cluster:**
- 3 nodes × (6 CPU, 24 GB RAM each)

**Job Set:**
- 12 jobs with HIGH/MEDIUM/LOW priorities
- HIGH: 5 jobs with tight deadlines (10-16 steps)
- MEDIUM: 4 jobs with moderate deadlines (18-21 steps)
- LOW: 3 jobs with loose deadlines (25-35 steps)

**Success Criteria:**
- Prioritize high-priority jobs → higher reward
- Maintain SLA compliance (60% weight)
- Optimize resource efficiency (30% weight)

**Expected Solution:**
- Priority queue: always schedule available high-priority jobs first
- Deadline-aware: schedule medium jobs to ensure completion
- Backfill with low-priority jobs

**Baseline Performance:** 0.75+ (with priority-aware scheduling)

---

### Hard Task: `schedule_with_dependencies`

**Scenario:** Dynamic job arrivals + job dependencies (DAG-structured workflows).

**Difficulty:** Jobs can only start after dependencies complete; must plan ahead.

**Cluster:**
- 3 nodes × (8 CPU, 32 GB RAM each)

**Job Set:**
- 8 jobs with dependencies:
  - Job 0 → Job 1 → Job 2 (chain)
  - Job 3 → Job 4 (chain, arrives at t=5)
  - Job 5 → Job 6 → Job 7 (chain, arrives at t=10)
- Dynamic arrivals at t=0, t=5, t=10

**Success Criteria:**
- Respect all dependencies (jobs cannot start until predecessors complete)
- SLA compliance (60% weight)
- Resource efficiency (30% weight)
- Dependency satisfaction (10% weight)

**Expected Solution:**
- Maintain dependency graph
- Greedy scheduling of available jobs (no unmet dependencies)
- Plan resource allocation to ensure all chains can complete
- Prioritize earlier chains to free up resources

**Baseline Performance:** 0.65+ (with dependency-aware DAG scheduling)

---

## 🚀 Setup & Usage

### Installation

```bash
git clone https://github.com/uday09012005/openenv-cloud-scheduler.git
cd openenv-cloud-scheduler
pip install -r requirements.txt
```

### Running Inference

Set environment variables:
```bash
export API_BASE_URL="http://localhost:8000/v1"  # Your LLM API endpoint
export MODEL_NAME="Qwen/Qwen2.5-7B"              # Your model
export HF_TOKEN="your_hf_token"                  # HuggingFace token
export TASK_ID="schedule_static_batch"           # Task to run
```

Run baseline:
```bash
python inference.py
```

### Running via OpenEnv

```bash
openenv validate openenv.yaml
openenv serve --host 0.0.0.0 --port 7860
```

### Docker Deployment (Hugging Face Spaces)

```bash
docker build -t openenv-cloud-scheduler .
docker run -p 7860:7860 \
  -e API_BASE_URL="http://localhost:8000/v1" \
  -e MODEL_NAME="gpt-4" \
  -e HF_TOKEN="$HF_TOKEN" \
  openenv-cloud-scheduler
```

---

## 📈 Evaluation Metrics

Each task is graded on a **normalized score ∈ [0.0, 1.0]** using task-specific graders:

### Easy Task Grader
```
Score = (completed_jobs / total_jobs) + (sla_compliance * 0.2) - (efficiency_penalty * 0.1)
```

### Medium Task Grader
```
Score = (sla_compliance * 0.7) + (resource_efficiency * 0.3)
```

### Hard Task Grader
```
Score = (sla_compliance * 0.6) + (resource_efficiency * 0.3) + (dependency_satisfaction * 0.1)
```

### Per-Episode Metrics
- **SLA Compliance:** Fraction of jobs meeting deadline
- **Resource Efficiency:** Utilization relative to total capacity
- **Makespan:** Total time to complete all jobs
- **Completion Count:** Number of jobs successfully scheduled and completed
- **Failed Count:** Number of jobs that missed deadline

---

## 🧠 Baseline Scores

| Task | Model | Score | SLA | Efficiency | Notes |
|------|-------|-------|-----|-----------|-------|
| `schedule_static_batch` | Random | 0.45 | 0.50 | 0.40 | Baseline random scheduling |
| `schedule_static_batch` | Greedy | 0.82 | 0.90 | 0.75 | Sort by deadline, fit-first |
| `schedule_with_priorities` | Random | 0.35 | 0.40 | 0.30 | Ignores priorities |
| `schedule_with_priorities` | Priority-Queue | 0.68 | 0.75 | 0.62 | Prioritizes high-priority jobs |
| `schedule_with_dependencies` | Random | 0.25 | 0.30 | 0.20 | Many dependency violations |
| `schedule_with_dependencies` | DAG-Aware | 0.58 | 0.65 | 0.50 | Respects dependencies, greedy scheduling |

---

## 🔧 Implementation Details

### Reward Function

The environment provides **rich, shaped rewards** (not sparse binary):

1. **Successful Completion:**
   - Base: +0.5 per job
   - Priority Bonus: +0.1 (medium), +0.2 (high)

2. **Deadline Missed:**
   - Penalty: -1.0 per job exceeding deadline

3. **Invalid Actions:**
   - Insufficient Resources: -0.3
   - Unmet Dependencies: -0.5
   - Infeasible Deadline: -0.2

4. **Progress Reward:**
   - Time Step: +0.05 (reward for making forward progress)

### Episode Termination

Episodes terminate when:
- All jobs completed or failed (clean termination)
- Max steps reached (`max_steps` varies by task: 100-150)
- No valid actions possible

---

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{openenv_cloud_scheduler_2024,
  title={Dynamic Cloud Job Scheduler: OpenEnv Benchmark},
  author={OpenEnv Contributors},
  year={2024},
  note={OpenEnv Hackathon Round 1}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please submit issues or PRs.

For questions or feedback, open an issue on GitHub.