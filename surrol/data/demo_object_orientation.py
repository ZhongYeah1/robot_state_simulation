#!/usr/bin/env python3
"""
演示如何在data_gen2.py中获取单臂四个任务的物体朝向数据

使用方法:
1. 直接运行此脚本查看各任务的物体朝向数据
2. 参考此脚本中的代码集成到您的数据收集流程中
"""
import os
import sys
import numpy as np
import gym
import pybullet as p

# 添加surrol路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.pybullet_utils import (
    get_all_object_info_in_camera, 
    robot_state_to_camera, 
    world_to_camera_position
)


def demonstrate_object_orientation_collection():
    """演示如何获取四个单臂任务的物体朝向数据"""
    
    # 四个单臂任务和对应的one-hot编码
    TASKS = [
        'NeedleReach-v0',    # [1,0,0]
        'GauzeRetrieve-v0',  # [0,1,0] 
        'NeedlePick-v0',     # [0,0,1]
        'PegTransfer-v0'     # [0,0,0]
    ]
    
    TASK_ENCODINGS = {
        'NeedleReach-v0':    [1, 0, 0],
        'GauzeRetrieve-v0':  [0, 1, 0],
        'NeedlePick-v0':     [0, 0, 1],
        'PegTransfer-v0':    [0, 0, 0]
    }
    
    for task_name in TASKS:
        print(f"\n{'='*50}")
        print(f"任务: {task_name}")
        print(f"{'='*50}")
        
        try:
            # 创建环境 - 使用DIRECT模式避免GUI连接冲突
            # render_mode=None 使PyBullet使用DIRECT模式（无GUI），避免多连接冲突
            env = gym.make(task_name, render_mode='human')  
            obs = env.reset()
            
            # 获取任务的one-hot编码
            task_encoding = TASK_ENCODINGS[task_name]
            print(f"任务编码: {task_encoding}")
            
            # 运行几步来展示数据收集
            for step in range(3):
                print(f"\n--- 步骤 {step + 1} ---")
                
                # 获取相机视图矩阵
                view_matrix = env.get_camera_params()
                
                # 1. 获取机器人状态 (7维: x,y,z,roll,pitch,yaw,jaw)
                if isinstance(obs, dict) and 'observation' in obs:
                    world_robot_state = obs['observation'][:7]
                    camera_robot_state = robot_state_to_camera(world_robot_state, view_matrix)
                    
                    print(f"机器人状态 (世界坐标系): {np.round(world_robot_state, 4)}")
                    print(f"机器人状态 (相机坐标系): {np.round(camera_robot_state, 4)}")
                
                # 2. 获取物体位置和朝向
                object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
                print(f"物体位置 (相机坐标系): {np.round(object_camera_pos, 4)}")
                print(f"物体朝向 (相机坐标系): {np.round(object_camera_ori, 4)}")
                
                # 3. 获取目标位置
                goal_world_pos = obs['desired_goal']
                goal_camera_pos = world_to_camera_position(goal_world_pos, view_matrix)
                print(f"目标位置 (相机坐标系): {np.round(goal_camera_pos, 4)}")
                
                # 4. 构造完整的19维状态向量
                # [7维机器人状态 + 3维任务编码 + 3维物体位置 + 3维物体朝向 + 3维目标位置]
                full_state_vector = (
                    camera_robot_state.tolist() +     # 7维机器人状态
                    task_encoding +                   # 3维任务编码  
                    object_camera_pos.tolist() +      # 3维物体位置
                    object_camera_ori.tolist() +      # 3维物体朝向 (新增!)
                    goal_camera_pos.tolist()          # 3维目标位置
                )
                
                print(f"完整状态向量维度: {len(full_state_vector)}")
                print(f"状态向量分解:")
                print(f"  机器人状态 (7维): {np.round(full_state_vector[0:7], 4)}")
                print(f"  任务编码 (3维):   {full_state_vector[7:10]}")
                print(f"  物体位置 (3维):   {np.round(full_state_vector[10:13], 4)}")
                print(f"  物体朝向 (3维):   {np.round(full_state_vector[13:16], 4)}  <-- 新增的朝向数据!")
                print(f"  目标位置 (3维):   {np.round(full_state_vector[16:19], 4)}")
                
                # 执行一个动作
                action = env.get_oracle_action(obs)
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
            
            # 关闭环境
            env.close()
            print(f"✓ 任务 {task_name} 演示完成")
            
            # 添加延迟确保完全关闭
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ 演示失败: {e}")
            import traceback
            traceback.print_exc()
            # 确保环境被关闭
            try:
                if 'env' in locals():
                    env.close()
            except:
                pass


def show_integration_example():
    """展示如何集成到data_gen2.py中"""
    
    print(f"\n{'='*60}")
    print("集成到 data_gen2.py 的示例代码:")
    print(f"{'='*60}")
    
    integration_code = '''
# 在 data_gen2.py 的 goToGoal 函数中，更新数据收集部分:

# 保存当前状态向量(相机坐标系)
if robot_states is not None and isinstance(obs, dict) and 'observation' in obs:
    if len(obs['observation']) >= 7:
        # 获取相机视图矩阵
        view_matrix = env.get_camera_params()

        # 1. 处理机器人状态 (前7维)
        world_state = obs['observation'][:7]
        camera_state = robot_state_to_camera(world_state, view_matrix)
        
        # 2. 获取物体的位置和朝向 (世界坐标系 -> 相机坐标系)
        object_camera_pos, object_camera_ori = get_all_object_info_in_camera(env, view_matrix)
        
        # 3. 目标点位置 (世界坐标系)
        goal_world_pos = obs['desired_goal']
        goal_camera_pos = world_to_camera_position(goal_world_pos, view_matrix)
        
        # 4. 组合成19维向量
        # [7维机器人状态 + 3维任务编码 + 3维物体位置 + 3维物体朝向 + 3维目标点位置]
        full_state = (camera_state.tolist() + 
                     task_encoding + 
                     object_camera_pos.tolist() + 
                     object_camera_ori.tolist() +    # 新增的物体朝向!
                     goal_camera_pos.tolist())
        
        robot_states.append(full_state)
'''
    
    print(integration_code)
    
    print("\n主要改进:")
    print("1. 新增了 get_all_object_info_in_camera() 函数来同时获取物体位置和朝向")
    print("2. 状态向量从16维扩展到19维，新增了3维物体朝向数据")
    print("3. 物体朝向以欧拉角形式表示 (roll, pitch, yaw)")
    print("4. 所有数据都已转换到相机坐标系，保持一致性")


if __name__ == "__main__":
    print("物体朝向数据获取功能演示")
    
    # 演示各个任务的物体朝向数据获取
    demonstrate_object_orientation_collection()
    
    # 展示集成示例
    show_integration_example()
    
    print(f"\n{'='*60}")
    print("演示完成!")
    print("您现在可以:")
    print("1. 运行修改后的 data_gen2.py 来收集包含物体朝向的数据")
    print("2. CSV文件现在包含19维状态向量 (之前是16维)")
    print("3. 新增的物体朝向数据位于索引 13-15 (roll, pitch, yaw)")
    print(f"{'='*60}")
