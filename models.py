from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class JobPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    ELEVATED = "elevated"
    HIGH = "high"

class Job(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    required_cpu: float = Field(..., description="Required CPU cores")
    required_ram: float = Field(..., description="Required RAM in GB")
    duration: int = Field(..., description="Job duration in time steps")
    priority: JobPriority = Field(default=JobPriority.LOW)
    deadline: int = Field(..., description="Absolute deadline")
    dependencies: List[str] = Field(default_factory=list)
    arrival_time: int = Field(default=0)

class Node(BaseModel):
    node_id: str
    total_cpu: float
    total_ram: float
    available_cpu: float
    available_ram: float
    running_jobs: List[str] = Field(default_factory=list)

class ClusterState(BaseModel):
    current_time: int
    nodes: List[Node]
    running_jobs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    completed_jobs: List[str] = Field(default_factory=list)
    failed_jobs: List[str] = Field(default_factory=list)
    pending_queue: List[Job] = Field(default_factory=list)

class Observation(BaseModel):
    cluster_state: ClusterState
    valid_actions: List[str]
    class Config:
        arbitrary_types_allowed = True

class Action(BaseModel):
    action_type: str
    job_id: Optional[str] = None
    node_id: Optional[str] = None

class Reward(BaseModel):
    value: float
    components: Dict[str, float] = Field(default_factory=dict)
    info: str = ""

class EpisodeResult(BaseModel):
    success: bool
    steps: int
    score: float
    rewards: List[float]
    sla_compliance: float
    resource_efficiency: float
    completed_count: int
    failed_count: int
    makespan: int
