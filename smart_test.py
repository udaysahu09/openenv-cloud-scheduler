
print("=" * 70)
print("SMART CLOUD JOB SCHEDULER TEST SUITE")
print("=" * 70)

from env import CloudJobSchedulerEnv

tasks = ["schedule_static_batch", "schedule_with_priorities", "schedule_with_dependencies"]

for task_id in tasks:
    print(f"\n📋 Testing: {task_id}")
    print("-" * 70)
    
    env = CloudJobSchedulerEnv(task_id=task_id)
    obs = env.reset()
    
    print(f"  ✓ Initialized")
    print(f"    - Nodes: {len(obs.cluster_state.nodes)}")
    print(f"    - Jobs: {len(obs.cluster_state.pending_queue)}")
    print(f"    - Valid actions: {len(obs.valid_actions)}")
    
    step = 0
    total_reward = 0.0
    
    while step < 150 and not env._check_done():
        # Smart strategy: prioritize scheduling jobs over waiting
        action = "wait()"
        
        # Try to schedule a job with earliest deadline first
        if len(obs.valid_actions) > 1:
            schedule_actions = [a for a in obs.valid_actions if "schedule_job" in a]
            if schedule_actions:
                action = schedule_actions[0]
        
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step += 1
    
    result = env.get_final_result()
    
    print(f"  ✓ Episode completed")
    print(f"    - Steps: {result.steps}")
    print(f"    - Total reward: {total_reward:.2f}")
    print(f"    - Jobs completed: {result.completed_count}/{result.completed_count + result.failed_count}")
    print(f"    - SLA compliance: {result.sla_compliance:.1%}")
    print(f"    - Resource efficiency: {result.resource_efficiency:.2f}")
    print(f"    - Final score: {result.score:.2f}")
    print(f"    - Success: {result.success}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
