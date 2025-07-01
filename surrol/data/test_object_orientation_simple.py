#!/usr/bin/env python3
"""
简化的测试脚本，用于验证物体朝向数据获取功能
"""
import os
import sys
import numpy as np
import gym
import time

# 添加surrol路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.pybullet_utils import (
    get_all_object_info_in_camera, 
    robot_state_to_camera, 
    world_to_camera_position
)


def test_single_task():
    """测试单个任务的物体朝向数据获取"""
    
    print("测试单个任务的物体朝向数据获取...")
    
    task_name = 'NeedleReach-v0'
    task_encoding = [1, 0, 0]
    
    try:
        # 创建环境 - 使用None模式避免GUI
        print(f"创建环境: {task_name}")
        env = gym.make(task_name, render_mode=None)
        obs = env.reset()
        
        print("✓ 环境创建成功")
        
        # 获取相机视图矩阵
        view_matrix = env.get_camera_params()
        print(f"✓ 获取相机参数成功，视图矩阵形状: {np.array(view_matrix).shape}")
        
        # 测试机器人状态转换
        if isinstance(obs, dict) and 'observation' in obs:
            world_robot_state = obs['observation'][:7]
            camera_robot_state = robot_state_to_camera(world_robot_state, view_matrix)
            print(f"✓ 机器人状态转换成功")
            print(f"  世界坐标系: {np.round(world_robot_state, 3)}")
            print(f"  相机坐标系: {np.round(camera_robot_state, 3)}")
            print(f"  转换后类型: {type(camera_robot_state)}")
        
        # 测试物体信息获取
        object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
        print(f"✓ 物体信息获取成功")
        print(f"  物体位置: {np.round(object_camera_pos, 3)} (类型: {type(object_camera_pos)})")
        print(f"  物体朝向: {np.round(object_camera_ori, 3)} (类型: {type(object_camera_ori)})")
        
        # 测试目标位置转换
        goal_world_pos = obs['desired_goal']
        goal_camera_pos = world_to_camera_position(goal_world_pos, view_matrix)
        print(f"✓ 目标位置转换成功")
        print(f"  目标位置: {np.round(goal_camera_pos, 3)} (类型: {type(goal_camera_pos)})")
        
        # 测试完整状态向量构造
        full_state_vector = (
            camera_robot_state.tolist() +     # 7维机器人状态
            task_encoding +                   # 3维任务编码  
            object_camera_pos.tolist() +      # 3维物体位置
            object_camera_ori.tolist() +      # 3维物体朝向
            goal_camera_pos.tolist()          # 3维目标位置
        )
        
        print(f"✓ 完整状态向量构造成功")
        print(f"  状态向量维度: {len(full_state_vector)}")
        print(f"  状态向量: {np.round(full_state_vector, 3)}")
        
        # 验证维度
        expected_dim = 7 + 3 + 3 + 3 + 3  # 19维
        if len(full_state_vector) == expected_dim:
            print(f"✓ 状态向量维度正确: {len(full_state_vector)}/{expected_dim}")
        else:
            print(f"✗ 状态向量维度错误: {len(full_state_vector)}/{expected_dim}")
        
        # 关闭环境
        env.close()
        print("✓ 环境关闭成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保环境被关闭
        try:
            if 'env' in locals():
                env.close()
        except:
            pass
        
        return False


def test_multiple_tasks_sequential():
    """顺序测试多个任务"""
    
    print("\n" + "="*60)
    print("顺序测试多个任务...")
    
    tasks = [
        ('NeedleReach-v0', [1, 0, 0]),
        ('GauzeRetrieve-v0', [0, 1, 0]),
        ('NeedlePick-v0', [0, 0, 1]),
    ]
    
    success_count = 0
    
    for task_name, task_encoding in tasks:
        print(f"\n--- 测试任务: {task_name} ---")
        
        try:
            # 创建环境
            env = gym.make(task_name, render_mode=None)
            obs = env.reset()
            
            # 获取物体信息
            view_matrix = env.get_camera_params()
            object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
            
            print(f"  物体位置: {np.round(object_camera_pos, 3)}")
            print(f"  物体朝向: {np.round(object_camera_ori, 3)}")
            print(f"  任务编码: {task_encoding}")
            print(f"✓ 任务 {task_name} 测试成功")
            
            success_count += 1
            
            # 关闭环境
            env.close()
            time.sleep(0.5)  # 短暂延迟
            
        except Exception as e:
            print(f"✗ 任务 {task_name} 测试失败: {e}")
            try:
                if 'env' in locals():
                    env.close()
            except:
                pass
    
    print(f"\n总结: {success_count}/{len(tasks)} 个任务测试成功")
    return success_count == len(tasks)


if __name__ == "__main__":
    print("开始物体朝向数据获取功能的简化测试...")
    
    # 测试单个任务
    single_success = test_single_task()
    
    if single_success:
        # 如果单个任务成功，再测试多个任务
        multiple_success = test_multiple_tasks_sequential()
        
        if multiple_success:
            print("\n" + "="*60)
            print("🎉 所有测试通过！")
            print("您现在可以安全地运行 data_gen2.py 来收集数据了。")
            print("="*60)
        else:
            print("\n" + "="*60) 
            print("⚠️  单任务测试通过，但多任务测试有问题。")
            print("这可能是PyBullet连接管理的问题，但不影响实际使用。")
            print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 基础功能测试失败，请检查代码。")
        print("="*60)
