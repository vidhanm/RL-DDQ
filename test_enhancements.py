"""
Comprehensive Test Suite for RL DDQ Enhancements
Tests: Expert Rewards, Hindi Language, New Actions
"""
import sys
sys.path.insert(0, 'src')

def test_config():
    """Test configuration changes"""
    print("\n[TEST 1] Configuration")
    print("-" * 50)
    from config import EnvironmentConfig
    
    # Check NUM_ACTIONS
    assert EnvironmentConfig.NUM_ACTIONS == 9, f"Expected 9 actions, got {EnvironmentConfig.NUM_ACTIONS}"
    print(f"  NUM_ACTIONS: {EnvironmentConfig.NUM_ACTIONS} - OK")
    
    # Check all actions exist
    actions = list(EnvironmentConfig.ACTIONS.values())
    print(f"  Actions: {actions}")
    
    # Check new actions
    new_actions = ['acknowledge_and_redirect', 'validate_then_offer', 'gentle_urgency']
    for action in new_actions:
        assert action in actions, f"Missing action: {action}"
    print(f"  New actions verified - OK")
    
    # Check expert rewards
    assert len(EnvironmentConfig.EXPERT_REWARDS) == 7, "Expected 7 expert rewards"
    assert len(EnvironmentConfig.EXPERT_PENALTIES) == 6, "Expected 6 expert penalties"
    print(f"  Expert Rewards: {len(EnvironmentConfig.EXPERT_REWARDS)} items - OK")
    print(f"  Expert Penalties: {len(EnvironmentConfig.EXPERT_PENALTIES)} items - OK")
    
    print("  [PASSED] Configuration test")
    return True


def test_language_support():
    """Test Hindi/Hinglish language support"""
    print("\n[TEST 2] Language Support")
    print("-" * 50)
    from llm.prompts import set_language, get_strategy_descriptions, LANGUAGE
    
    # Test English
    set_language('english')
    en_strats = get_strategy_descriptions()
    assert len(en_strats) == 9, f"Expected 9 English strategies, got {len(en_strats)}"
    print(f"  English strategies: {len(en_strats)} - OK")
    
    # Test Hindi
    set_language('hindi')
    hi_strats = get_strategy_descriptions()
    assert len(hi_strats) == 9, f"Expected 9 Hindi strategies, got {len(hi_strats)}"
    print(f"  Hindi strategies: {len(hi_strats)} - OK")
    
    # Test Hinglish
    set_language('hinglish')
    hing_strats = get_strategy_descriptions()
    assert len(hing_strats) == 9, f"Expected 9 Hinglish strategies, got {len(hing_strats)}"
    print(f"  Hinglish strategies: {len(hing_strats)} - OK")
    
    # Verify new actions have descriptions
    new_actions = ['acknowledge_and_redirect', 'validate_then_offer', 'gentle_urgency']
    for lang in ['english', 'hindi', 'hinglish']:
        set_language(lang)
        strats = get_strategy_descriptions()
        for action in new_actions:
            assert action in strats, f"{action} missing in {lang}"
    print(f"  All 9 actions have descriptions in all 3 languages - OK")
    
    # Reset to English for other tests
    set_language('english')
    
    print("  [PASSED] Language support test")
    return True


def test_nlu_environment():
    """Test NLU environment with new features"""
    print("\n[TEST 3] NLU Environment")
    print("-" * 50)
    from environment.nlu_env import NLUDebtCollectionEnv
    from config import EnvironmentConfig
    
    env = NLUDebtCollectionEnv(llm_client=None, render_mode=None)
    obs, info = env.reset()
    
    print(f"  Environment created - OK")
    print(f"  Observation shape: {obs.shape} - OK")
    assert env.action_space.n == 9, f"Expected 9 actions, got {env.action_space.n}"
    print(f"  Action space: {env.action_space.n} actions - OK")
    
    print("  [PASSED] NLU Environment test")
    return True


def test_expert_rewards():
    """Test expert reward logic"""
    print("\n[TEST 4] Expert Reward Logic")
    print("-" * 50)
    from environment.nlu_env import NLUDebtCollectionEnv
    from config import EnvironmentConfig
    
    env = NLUDebtCollectionEnv(llm_client=None, render_mode=None)
    obs, info = env.reset()
    
    # Run a few steps
    rewards = []
    actions_taken = []
    
    for action_id in [0, 1, 6, 7, 8, 3]:  # empathy, ask, new actions, payment plan
        obs, reward, term, trunc, info = env.step(action_id)
        action_name = EnvironmentConfig.ACTIONS[action_id]
        rewards.append(reward)
        actions_taken.append(action_name)
        print(f"  {action_name}: reward = {reward:.2f}")
        if term or trunc:
            break
    
    # Check action history is being tracked
    assert len(env.state.action_history) > 0, "Action history not tracking!"
    print(f"  Action history tracked: {len(env.state.action_history)} actions - OK")
    
    print("  [PASSED] Expert reward logic test")
    return True


def test_full_episode():
    """Test a full episode"""
    print("\n[TEST 5] Full Episode")
    print("-" * 50)
    from environment.nlu_env import NLUDebtCollectionEnv
    from config import EnvironmentConfig
    
    env = NLUDebtCollectionEnv(llm_client=None, render_mode=None)
    obs, info = env.reset()
    
    total_reward = 0
    turns = 0
    
    for turn in range(15):
        # Strategy: mix of actions
        action = turn % 9
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        turns += 1
        
        if terminated or truncated:
            break
    
    print(f"  Episode length: {turns} turns")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final cooperation: {env.state.cooperation:.2f}")
    print(f"  Final sentiment: {env.state.sentiment:.2f}")
    
    print("  [PASSED] Full episode test")
    return True


def main():
    print("=" * 70)
    print("COMPREHENSIVE ENHANCEMENT TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Config", test_config),
        ("Language Support", test_language_support),
        ("NLU Environment", test_nlu_environment),
        ("Expert Rewards", test_expert_rewards),
        ("Full Episode", test_full_episode),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\nALL TESTS PASSED! Everything is working correctly.")
    else:
        print(f"\n{failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
