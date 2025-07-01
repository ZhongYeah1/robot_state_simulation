#!/usr/bin/env python3
"""
测试物体朝向数据获取功能
"""
import os
import sys
import numpy as np
import gym
import pybullet as p

# 添加surrol路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.pybullet_utils import get_all_object_info_in_camera, robot_state_to_camera


def test_object_orientation_collection():
    """测试各个任务的物体朝向数据收集"""
    
    # 定义测试任务
    test_tasks = [
        'NeedleReach-v0',
        'GauzeRetrieve-v0', 
        'NeedlePick-v0',
        'PegTransfer-v0'
    ]
    
    for task_name in test_tasks:
        print(f"\n=== 测试任务: {task_name} ===")
        
        try:
            # 创建环境
            env = gym.make(task_name, render_mode='human')
            obs = env.reset()
            
            # 获取相机视图矩阵
            view_matrix = env.get_camera_params()
            
            # 测试物体信息获取
            object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
            
            print(f"物体位置 (相机坐标系): {object_camera_pos}")
            print(f"物体朝向 (相机坐标系): {object_camera_ori}")
            
            # 测试机器人状态转换
            if isinstance(obs, dict) and 'observation' in obs:
                if len(obs['observation']) >= 7:
                    world_state = obs['observation'][:7]
                    camera_state = robot_state_to_camera(world_state, view_matrix)
                    print(f"机器人状态 (世界坐标系): {world_state}")
                    print(f"机器人状态 (相机坐标系): {camera_state}")
            
            # 测试完整状态向量构造
            task_encoding = [1, 0, 0] if 'NeedleReach' in task_name else \
                          [0, 1, 0] if 'GauzeRetrieve' in task_name else \
                          [0, 0, 1] if 'NeedlePick' in task_name else [0, 0, 0]
            
            goal_world_pos = obs['desired_goal']
            from utils.pybullet_utils import world_to_camera_position
            goal_camera_pos = world_to_camera_position(goal_world_pos, view_matrix)
            
            # 构造19维状态向量
            full_state = (camera_state.tolist() + 
                         task_encoding + 
                         object_camera_pos.tolist() + 
                         object_camera_ori.tolist() + 
                         goal_camera_pos.tolist())
            
            print(f"完整状态向量维度: {len(full_state)}")
            print(f"完整状态向量: {np.round(full_state, 4)}")
            
            # 验证维度
            assert len(full_state) == 19, f"状态向量维度错误: 期望19维，实际{len(full_state)}维"
            print("✓ 状态向量维度正确")
            
            # 关闭环境
            env.close()
            print("✓ 测试通过")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()


def test_single_task_data_collection():
    """测试单个任务的数据收集流程"""
    
    print("\n=== 测试单个任务数据收集流程 ===")
    
    task_name = 'NeedleReach-v0'
    env = gym.make(task_name, render_mode='human')
    
    try:
        obs = env.reset()
        
        # 模拟数据收集过程
        robot_states = []
        task_encoding = [1, 0, 0]  # NeedleReach的编码
        
        for step in range(5):  # 收集5步数据
            # 获取相机视图矩阵
            view_matrix = env.get_camera_params()
            
            # 获取机器人状态
            world_state = obs['observation'][:7]
            camera_state = robot_state_to_camera(world_state, view_matrix)
            
            # 获取物体信息
            object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
            
            # 获取目标位置
            goal_world_pos = obs['desired_goal']
            from utils.pybullet_utils import world_to_camera_position
            goal_camera_pos = world_to_camera_position(goal_world_pos, view_matrix)
            
            # 构造完整状态向量
            full_state = (camera_state.tolist() + 
                         task_encoding + 
                         object_camera_pos.tolist() + 
                         object_camera_ori.tolist() + 
                         goal_camera_pos.tolist())
            
            robot_states.append(full_state)
            
            # 执行一步
            action = env.get_oracle_action(obs)
            obs, reward, done, info = env.step(action)
            
            print(f"步骤 {step+1}: 状态向量维度 = {len(full_state)}")
        
        print(f"✓ 成功收集了 {len(robot_states)} 个状态向量")
        print(f"✓ 每个状态向量的维度: {len(robot_states[0])}")
        
        # 检查状态向量的变化
        if len(robot_states) > 1:
            state_diff = np.array(robot_states[1]) - np.array(robot_states[0])
            print(f"✓ 状态向量在变化 (最大变化量: {np.max(np.abs(state_diff)):.6f})")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    print("开始测试物体朝向数据获取功能...")
    
    # 测试各个任务的物体朝向获取
    test_object_orientation_collection()
    
    # 测试单个任务的完整数据收集流程
    test_single_task_data_collection()
    
    print("\n测试完成!")
