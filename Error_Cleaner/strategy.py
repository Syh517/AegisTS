import numpy as np
import pandas as pd
import time

def save_trained_agents(trained_agents, filepath):
    """
    保存训练好的代理模型到文件
    
    参数:
    - trained_agents: 训练好的代理模型 (macro_agent, low_agent)
    - filepath: 保存文件路径
    """
    import pickle
    import os

    macro_agent, low_agent = trained_agents
    
    # 创建保存目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存模型
    with open(filepath, 'wb') as f:
        pickle.dump({
            'macro_agent_q': macro_agent.q,
            'low_agent_q': low_agent.q,
            'high_actions': macro_agent.actions,
            'low_actions_map': low_agent.low_actions_map,
            'macro_agent_params': {
                'lr': macro_agent.lr,
                'gamma': macro_agent.gamma,
                'eps_start': macro_agent.eps_start,
                'eps_end': macro_agent.eps_end,
                'eps_decay': macro_agent.eps_decay
            },
            'low_agent_params': {
                'lr': low_agent.lr,
                'gamma': low_agent.gamma,
                'eps_start': low_agent.eps_start,
                'eps_end': low_agent.eps_end,
                'eps_decay': low_agent.eps_decay
            }
        }, f)
    
    print(f"训练好的策略模型已保存到: {filepath}")

def load_trained_agents(filepath, state_h_dim, state_l_dim):
    """
    从文件加载训练好的代理模型
    
    参数:
    - filepath: 保存文件路径
    - state_h_dim: 高层状态维度
    - state_l_dim: 低层状态维度
    
    返回:
    - trained_agents: 训练好的代理模型 (macro_agent, low_agent)
    """
    import pickle
    import os
    from Error_Cleaner.RLclean import MacroAgent, LowAgent

    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到模型文件: {filepath}")
    
    # 加载模型
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 重建代理模型
    macro_agent = MacroAgent(
        actions=data['high_actions'],
        state_dim=state_h_dim,
        **data['macro_agent_params']
    )
    
    low_agent = LowAgent(
        low_actions_map=data['low_actions_map'],
        state_dim=state_l_dim,
        **data['low_agent_params']
    )
    
    # 恢复Q表
    macro_agent.q = data['macro_agent_q']
    low_agent.q = data['low_agent_q']
    
    print(f"训练好的策略模型已从 {filepath} 加载")
    return macro_agent, low_agent

def save_single_layer_agent(agent, filepath):
    """
    保存训练好的单层代理模型到文件
    
    参数:
    - agent: 训练好的单层代理模型
    - filepath: 保存文件路径
    """
    import pickle
    import os

    # 创建保存目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存模型
    with open(filepath, 'wb') as f:
        pickle.dump({
            'agent_q': agent.q,
            'all_actions': agent.all_actions,
            'agent_params': {
                'lr': agent.lr,
                'gamma': agent.gamma,
                'eps_start': agent.eps_start,
                'eps_end': agent.eps_end,
                'eps_decay': agent.eps_decay
            },
            'agent_steps': agent.steps
        }, f)
    
    print(f"单层代理模型已保存到: {filepath}")

def load_single_layer_agent(filepath, state_dim, all_actions):
    """
    从文件加载训练好的单层代理模型
    
    参数:
    - filepath: 保存文件路径
    - state_dim: 状态维度
    - all_actions: 所有可能的操作列表
    
    返回:
    - agent: 训练好的单层代理模型
    """
    import pickle
    import os
    from Error_Cleaner.single_layer import SingleLayerAgent

    # 检查文件是否存在
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到模型文件: {filepath}")
    
    # 加载模型
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 重建代理模型
    agent = SingleLayerAgent(
        all_actions=all_actions,
        state_dim=state_dim,
        lr=data['agent_params']['lr'],
        gamma=data['agent_params']['gamma'],
        eps_start=data['agent_params']['eps_start'],
        eps_end=data['agent_params']['eps_end'],
        eps_decay=data['agent_params']['eps_decay']
    )
    
    # 恢复Q表
    agent.q = data['agent_q']
    agent.steps = data.get('agent_steps', 0)  # 如果旧模型没有保存steps，则默认为0
    
    print(f"单层代理模型已从 {filepath} 加载")
    return agent


def continue_training_from_saved_model(
    model_path, 
    dirty_data,
    label, 
    task_type, 
    detector, 
    constraints, 
    additional_episodes=10,
    concentrated=True,
    max_steps=20
):
    """
    从已保存的模型继续训练
    
    参数:
    - model_path: 已保存模型的路径
    - dirty_data: 待清洗的数据
    - label: 标签数据
    - task_type: 任务类型 ('classification', 'forecast', 'clustering')
    - detector: 检测器
    - constraints: 约束条件
    - additional_episodes: 额外的训练轮数
    - concentrated: 是否使用统一约束
    - max_steps: 最大步数
    
    返回:
    - repaired_data: 修复后的数据
    - updated_agents: 更新后的代理模型
    """
    from Error_Cleaner.RLclean import RLCleanEnvironment, train_hrl_agents
    from Error_Cleaner.RLclean import train_and_evaluate
    
    # 创建环境以获取状态维度
    temp_env = RLCleanEnvironment(
        dirty_data=dirty_data,
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 获取状态维度
    state_h_dim = temp_env._state_high().shape[0]
    state_l_dim = temp_env._state_low('MISSING').shape[0]
    
    # 关闭临时环境
    del temp_env
    
    # 加载已保存的模型，使用正确的状态维度
    loaded_macro_agent, loaded_low_agent = load_trained_agents(
        model_path, 
        state_h_dim=state_h_dim,
        state_l_dim=state_l_dim
    )
    
    # 创建环境用于继续训练
    env = RLCleanEnvironment(
        dirty_data=dirty_data,
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 继续训练
    print(f"\n--- 从保存的模型继续训练 {additional_episodes} 轮")
    high_reward_history = train_hrl_agents(env, loaded_macro_agent, loaded_low_agent, n_episodes=additional_episodes)
    print("--- 继续训练完成 ---")
    
    # 评估最终性能
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    # 计算总时间成本
    total_time_cost = 0  # 在继续训练的情况下，时间成本可以不计入初始训练时间
    print(f"额外训练时间成本: {total_time_cost:.2f} 秒")
    
    # 返回清洗后的数据、时间成本以及训练好的策略模型
    return env.cur_train, total_time_cost, (loaded_macro_agent, loaded_low_agent)


def apply_cleaning_from_saved_model(
    model_path, 
    dirty_data, 
    label, 
    task_type, 
    detector, 
    constraints, 
    concentrated=True,
    max_steps=20
):
    """
    使用已保存的模型对脏数据进行清洗
    
    参数:
    - model_path: 已保存模型的路径
    - dirty_data: 待清洗的数据
    - label: 标签数据
    - task_type: 任务类型 ('classification', 'forecast', 'clustering')
    - detector: 检测器
    - constraints: 约束条件
    - concentrated: 是否使用统一约束
    - max_steps: 最大步数
    
    返回:
    - cleaned_data: 清洗后的数据
    """
    from Error_Cleaner.RLclean import RLCleanEnvironment, macro_agent_choose_with_rates, low_agent_choose_override
    from Error_Cleaner.RLclean import train_and_evaluate
    
    # 创建环境以获取状态维度
    temp_env = RLCleanEnvironment(
        dirty_data=dirty_data, 
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 获取状态维度
    state_h_dim = temp_env._state_high().shape[0]
    state_l_dim = temp_env._state_low('MISSING').shape[0]
    
    # 关闭临时环境
    del temp_env
    
    # 加载已保存的模型，使用正确的状态维度
    loaded_macro_agent, loaded_low_agent = load_trained_agents(
        model_path, 
        state_h_dim=state_h_dim,
        state_l_dim=state_l_dim
    )
    
    # 创建环境用于应用清洗策略
    env = RLCleanEnvironment(
        dirty_data=dirty_data,
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 重置环境状态
    state_h = env.reset()
    done = False
    step_count = 0

    start_time = time.time()
    print(f"\n--- 应用已保存的模型进行数据清洗 ---")

    while not done and step_count < max_steps:
        rates, _, _, _, _, _ = env._get_issue_rates()
        macro_action = macro_agent_choose_with_rates(loaded_macro_agent, state_h, rates, env)
        
        if step_count >= max_steps - 1:
            macro_action = 'FINISH'

        if macro_action == 'FINISH':
            env.step_count += 1
            next_state_h, reward_h, done, info = env.step_high(macro_action, 0.0, is_final=True)
            print(f"清洗完成 - 最终性能: {info.get('final_perf', 0.0):.4f}")
            break

        state_l = env._state_low(macro_action)
        # 使用重写的选择函数
        low_action = low_agent_choose_override(loaded_low_agent, state_l, macro_action, env)
        next_state_l, reward_l, low_done = env.step_low(macro_action, low_action)
        low_trajectory = [(state_l, low_action, reward_l, next_state_l, False)]
        
        start_state_h = state_h
        next_state_h, reward_h, high_done, info = env.step_high(macro_action, reward_l)
        
        # 记录动作和性能变化
        env.action_history.append((macro_action, low_action))
        test_perf_change = info.get('test_perf_change', 0.0)
        env.performance_history.append(test_perf_change)
        
        # 更新动作计数
        if macro_action in env.action_count:
            env.action_count[macro_action] += 1
        else:
            env.action_count[macro_action] = 1
            
        # 更新动作效果记录
        env.recent_action_effects[macro_action] = test_perf_change
        
        # 保持历史记录长度合理
        if len(env.action_history) > 10:
            env.action_history.pop(0)
            env.performance_history.pop(0)
        
        done = low_done or high_done
        state_h = next_state_h
        step_count += 1

        test_change = info.get('test_perf_change', 0.0)
        extra_penalty = info.get('extra_penalty', 0.0)
        eval_reward = info.get('eval_reward', 0.0)
        print(f"Step {step_count}: 高动作={macro_action:<10} "
              f"低动作={low_action:<20} "
              f"奖励={reward_h:+6.3f} "
              f"测试性能={info.get('test_perf', 0):.4f} "
              f"ΔTest={test_change:+.4f} "
              f"EvalReward={eval_reward:+.4f} "
              f"Penalty={extra_penalty:+.4f}")

    print("--- 数据清洗完成 ---")
    total_time_cost = time.time() - start_time

    # 评估最终性能
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")

    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    # 返回清洗后的数据
    return env.cur_train, total_time_cost


def continue_training_single_layer_model(
    model_path, 
    dirty_data,
    label, 
    task_type, 
    detector, 
    constraints, 
    additional_episodes=10,
    concentrated=True,
    max_steps=20
):
    """
    从已保存的单层模型继续训练
    
    参数:
    - model_path: 已保存模型的路径
    - dirty_data: 待清洗的数据
    - label: 标签数据
    - task_type: 任务类型 ('classification', 'forecast', 'clustering')
    - detector: 检测器
    - constraints: 约束条件
    - additional_episodes: 额外的训练轮数
    - concentrated: 是否使用统一约束
    - max_steps: 最大步数
    
    返回:
    - repaired_data: 修复后的数据
    - updated_agent: 更新后的代理模型
    """
    from Error_Cleaner.single_layer import RLCleanEnvironment, train_single_layer_agent, get_all_possible_actions
    from Error_Cleaner.RLclean import train_and_evaluate
    
    # 创建环境以获取状态维度
    temp_env = RLCleanEnvironment(
        dirty_data=dirty_data, 
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 获取状态维度和所有可能的操作
    state_dim = temp_env._state_high().shape[0]
    all_actions = get_all_possible_actions()
    
    # 关闭临时环境
    del temp_env
    
    # 加载已保存的单层模型，使用正确的状态维度和动作空间
    loaded_agent = load_single_layer_agent(
        model_path, 
        state_dim=state_dim,
        all_actions=all_actions
    )
    
    # 创建环境用于继续训练
    env = RLCleanEnvironment(
        dirty_data=dirty_data, 
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 继续训练
    print(f"\n--- 从保存的单层模型继续训练 {additional_episodes} 轮")
    reward_history = train_single_layer_agent(env, loaded_agent, n_episodes=additional_episodes)
    print("--- 继续训练完成 ---")
    
    # 评估最终性能
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    # 计算总时间成本
    total_time_cost = 0  # 在继续训练的情况下，时间成本可以不计入初始训练时间
    print(f"额外训练时间成本: {total_time_cost:.2f} 秒")
    
    # 返回清洗后的数据、时间成本以及训练好的策略模型
    return env.cur_train, total_time_cost, loaded_agent


def apply_single_layer_model(dirty_data, label, task_type, detector, constraints, model_path, concentrated=True, max_steps=20):
    """
    使用训练好的单层模型对新数据进行清洗
    
    参数:
    - dirty_data: 需要清洗的脏数据
    - label: 数据标签（如果是分类任务）
    - task_type: 任务类型 ('classification', 'forecast', 'clustering')
    - detector: 检测器实例
    - constraints: 约束条件
    - model_path: 保存的模型路径
    - concentrated: 是否使用集中式约束
    - max_steps: 最大清洗步骤数
    
    返回:
    - cleaned_data: 清洗后的数据
    """
    from Error_Cleaner.single_layer import RLCleanEnvironment, get_all_possible_actions
    from Error_Cleaner.single_layer import single_agent_choose_with_repeat_check, train_and_evaluate

    # 创建环境以获取状态维度
    temp_env = RLCleanEnvironment(
        dirty_data=dirty_data,
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )
    
    # 获取状态维度和所有可能的操作
    state_dim = temp_env._state_high().shape[0]
    all_actions = get_all_possible_actions()
    
    # 关闭临时环境
    del temp_env

    # 加载已保存的模型，使用正确的状态维度和动作空间
    loaded_agent = load_single_layer_agent(
        model_path, 
        state_dim=state_dim,
        all_actions=all_actions
    )

    # 创建一个新的环境用于清洗
    env = RLCleanEnvironment(
        dirty_data=dirty_data, 
        label=label, 
        task_type=task_type, 
        detector=detector, 
        constraints=constraints, 
        concentrated=concentrated, 
        max_steps=max_steps, 
        train_ratio=0.7
    )

    # 重置环境状态
    state = env.reset()
    done = False
    step_count = 0

    start_time = time.time()
    print("\n--- 开始应用单层模型清洗策略 ---")

    # 使用训练好的策略进行清洗
    while not done and step_count < max_steps:
        # 选择动作
        action = single_agent_choose_with_repeat_check(loaded_agent, state, env)
        high_action, low_action = action

        if step_count >= max_steps - 1:
            high_action = 'FINISH'
            low_action = 'NONE'
            action = ('FINISH', 'NONE')

        if high_action == 'FINISH':
            env.step_count += 1
            next_state, reward, done, info = env.step_high(high_action, 0.0, is_final=True)
            print(f"清洗完成 - 最终性能: {info.get('final_perf', 0.0):.4f}")
            break

        # 执行低层动作
        next_state_l = env._state_low(high_action)
        # 执行实际的清洗操作
        next_state_l, reward_l, low_done = env.step_low(high_action, low_action)

        # 执行高层动作
        start_state = state
        next_state, reward_h, high_done, info = env.step_high(high_action, reward_l)

        # 记录动作和性能变化
        env.action_history.append((high_action, low_action))
        test_perf_change = info.get('test_perf_change', 0.0)
        env.performance_history.append(test_perf_change)

        # 更新动作计数
        action_tuple = (high_action, low_action)
        if action_tuple in env.action_count:
            env.action_count[action_tuple] += 1
        else:
            env.action_count[action_tuple] = 1

        # 更新动作效果记录
        env.recent_action_effects[action_tuple] = test_perf_change

        # 保持历史记录长度合理
        if len(env.action_history) > 10:
            env.action_history.pop(0)
            env.performance_history.pop(0)

        done = low_done or high_done
        state = next_state
        step_count += 1

        test_change = info.get('test_perf_change', 0.0)
        extra_penalty = info.get('extra_penalty', 0.0)
        eval_reward = info.get('eval_reward', 0.0)
        print(f"Step {step_count}: 动作=({high_action}, {low_action}), "
              f"奖励={reward_h:.4f}, 测试性能={info.get('test_perf', 0):.4f}, "
              f"ΔTest={test_change:+.4f}, EvalReward={eval_reward:+.4f}, "
              f"Penalty={extra_penalty:+.4f}")

    print("--- 单层模型清洗策略应用完成 ---\n")
    total_time_cost = time.time() - start_time

    # 评估最终性能
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")
    
    final_test_perf, _ = train_and_evaluate(task_type, 'final', env.final_model, env)
    print(f"\n【最终】深度学习模型性能: {final_test_perf:.4f}")
    print(f"【初始】代理模型性能: {env.initial_proxy_perf:.4f}")
    print(f"【初始】深度学习模型性能: {env.initial_final_perf:.4f}")
    print(f"最终最佳簇数: {env.best_k}")
    print("\n")

    return env.cur_train, total_time_cost
