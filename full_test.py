print("=" * 70)
print("COMPLETE CLOUD JOB SCHEDULER TEST SUITE")
print("=" * 70)

from env import CloudJobSchedulerEnv

# Test all 3 tasks
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
    
    # Run episode
    step = 0
    total_reward = 0.0
    
    while step < 100 and not env._check_done():
        action = obs.valid_actions[0] if obs.valid_actions else "wait()"
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step += 1
    
    result = env.get_final_result()
    
    print(f"  ✓ Episode completed")
    print(f"    - Steps: {result.steps}")
    print(f"    - Reward: {total_reward:.2f}")
    print(f"    - Jobs completed: {result.completed_count}/{result.completed_count + result.failed_count}")
    print(f"    - SLA compliance: {result.sla_compliance:.1%}")
    print(f"    - Final score: {result.score:.2f}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)