print("Starting tests...")

from models import Job
from env import CloudJobSchedulerEnv

env = CloudJobSchedulerEnv()
obs = env.reset()

print("✓ Environment works!")
print(f"Nodes: {len(obs.cluster_state.nodes)}")
print(f"Jobs: {len(obs.cluster_state.pending_queue)}")

print("\n✅ TEST PASSED!")
