"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
import csv
from PIL import Image
from surrol.const import ROOT_DIR_PATH

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
# parser.add_argument('--env', type=str, required=True,
#                     help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

actions = []
observations = []
infos = []

images = []  # record video
masks = []
video_index = 0 

# 定义要收集的任务和对应的one-hot编码
TASKS = [
    'NeedleReach-v0',    # 索引 0-99
    'GauzeRetrieve-v0',  # 索引 100-199
    'NeedlePick-v0',     # 索引 200-299
    'PegTransfer-v0'     # 索引 300-399
]

TASK_ENCODINGS = {
    'NeedleReach-v0':    [1, 0, 0],
    'GauzeRetrieve-v0':  [0, 1, 0],
    'NeedlePick-v0':     [0, 0, 1],
    'PegTransfer-v0':    [0, 0, 0]
}

# 每个任务收集的样本数量
SAMPLES_PER_TASK = 100


def main():
    global video_index
    
    # 创建存储目录
    base_folder = os.path.join(ROOT_DIR_PATH, 'data')
    video_base_folder = os.path.join(base_folder, 'video')  # 视频每帧
    film_base_folder = os.path.join(base_folder, 'film')    # 视频
    label_folder = os.path.join(base_folder, 'label')
    label_npz_folder = os.path.join(base_folder, 'label_npz')
    
    os.makedirs(video_base_folder, exist_ok=True)
    os.makedirs(film_base_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(label_npz_folder, exist_ok=True)

    # 开始收集所有任务数据
    for task_idx, task_name in enumerate(TASKS):
        print(f"\n\n===== 开始收集任务 {task_name} 的数据 =====\n")
        
        # 设置视频索引的起始值
        video_index = task_idx * SAMPLES_PER_TASK
        start_index = video_index
        
        # 为当前任务创建环境
        env = gym.make(task_name, render_mode='human')
        
        # 打印observation space和action space
        print("\nObservation Space:")
        print(f"类型: {type(env.observation_space)}")
        print(f"结构: {env.observation_space}")
        
        print("\nAction Space:")
        print(f"类型: {type(env.action_space)}")
        print(f"结构: {env.action_space}")
        print(f"动作范围: {env.action_space.low} 到 {env.action_space.high}" 
              if hasattr(env.action_space, 'low') else "")
        
        # 如果是Dict观察空间，打印更详细信息
        if hasattr(env.observation_space, 'spaces'):
            print("\n观察空间详情:")
            for key, space in env.observation_space.spaces.items():
                print(f"  {key}: {type(space)}, 形状: {space.shape}")
        
        # 为当前任务重置变量
        actions = []
        observations = []
        infos = []
        masks = []
        
        num_itr = SAMPLES_PER_TASK
        cnt = 0
        init_state_space = 'random'
        env.reset()
        print("Reset!")
        init_time = time.time()
        
        if args.steps is None:
            args.steps = env._max_episode_steps
        
        print()
        while len(actions) < num_itr:
            obs = env.reset()
            print(f"任务: {task_name} - ITERATION NUMBER {len(actions)}")
            
            # 清空图像列表，准备收集新一轮的图像
            current_images = []
            
            # 创建视频文件夹
            if args.video:
                video_folder = os.path.join(video_base_folder, f"video_{video_index}")
                os.makedirs(video_folder, exist_ok=True)
                print(f"视频序列 {video_index} 将保存到: {video_folder}")
            
            # 记录当前视频序列对应的状态向量
            robot_states = []
            
            # 获取当前任务的one-hot编码
            task_encoding = TASK_ENCODINGS[task_name]
            
            # 收集轨迹
            goToGoal(env, obs, robot_states, current_images, task_encoding)
            
            # 如果启用了视频录制并成功完成了任务
            if args.video and len(actions) > cnt:  # 说明任务成功了
                # 保存这个序列的每一帧图像
                for i, img in enumerate(current_images):
                    img_filename = os.path.join(video_folder, f"img_{i}.png")
                    Image.fromarray(img).save(img_filename)
                
                # 保存对应的状态向量到CSV (已包含one-hot编码)
                csv_filename = os.path.join(label_folder, f"label_{video_index}.csv")
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # 写入状态向量数据（已在goToGoal中添加了one-hot编码）
                    for state in robot_states:
                        writer.writerow(state)
                
                # 创建并保存视频文件
                film_filename = os.path.join(film_base_folder, f"video_{video_index}.mp4")
                writer = imageio.get_writer(film_filename, fps=20)
                for img in current_images:
                    writer.append_data(img)
                writer.close()
                
                print(f"保存了 {len(current_images)} 帧图像和 {len(robot_states)} 个状态向量")
                print(f"图像保存到: {video_folder}")
                print(f"视频保存到: {film_filename}")
                print(f"状态向量保存到: {csv_filename}")
                if len(current_images) != len(robot_states):
                    print(f"警告：图像({len(current_images)})和状态向量({len(robot_states)})数量不匹配!")
                
                # 增加视频索引，为下一个视频序列做准备
                video_index += 1
            
            cnt += 1
        
        # 保存当前任务的数据到npz文件
        file_name = f"data_{task_name}_{init_state_space}_{num_itr}_{start_index}-{video_index-1}.npz"
        folder = 'demo' if not args.video else 'label_npz'
        folder = os.path.join(base_folder, folder)
        
        np.savez_compressed(os.path.join(folder, file_name),
                            acs=actions, obs=observations, info=infos)
        
        if args.video and len(masks) > 0:
            mask_name = f"mask_{task_name}.npz"
            np.savez_compressed(os.path.join(folder, mask_name), masks=masks)
        
        used_time = time.time() - init_time
        print(f"任务 {task_name} 数据保存在: {folder}")
        print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
        print(f"Trials: {num_itr}/{cnt}")
        
        # 关闭当前环境
        env.close()


def goToGoal(env, last_obs, robot_states=None, current_images=None, task_encoding=None):
    episode_acs = []
    episode_obs = []
    episode_info = []
    
    time_step = 0  # count the total number of time steps
    episode_init_time = time.time()
    episode_obs.append(last_obs)
    # last_obs不放到csv里，因为图像没有初始状态
    obs, success = last_obs, False
    
    while time_step < min(env._max_episode_steps, args.steps):
        action = env.get_oracle_action(obs)
        
        # img和obs一一对应
        if args.video and current_images is not None:
            # img, mask = env.render('img_array')
            img = env.render('rgb_array')
            current_images.append(img)
            # masks.append(mask)
            
            # 保存当前状态向量(相机坐标系)
            if robot_states is not None and isinstance(obs, dict) and 'observation' in obs:
                if len(obs['observation']) >= 7:
                    world_state = obs['observation'][:7]
                    
                    # 获取相机视图矩阵并转换到相机坐标系
                    view_matrix = env.get_camera_params()
                    from surrol.utils.pybullet_utils import robot_state_to_camera
                    camera_state = robot_state_to_camera(world_state, view_matrix)
                    
                    # 添加任务特定的one-hot编码，扩展为10维向量
                    full_state = camera_state.tolist() + task_encoding
                    
                    robot_states.append(full_state)
        
        obs, reward, done, info = env.step(action)
        
        print(f" -> robot_state: {obs['observation'][:7]}, \nreward: {reward}\n")
        
        time_step += 1
        
        if isinstance(obs, dict) and info['is_success'] > 0 and not success:
            print("Timesteps to finish:", time_step)
            success = True
        
        episode_acs.append(action)
        episode_info.append(info)
        episode_obs.append(obs)
    print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))
    
    if success:
        actions.append(episode_acs)
        observations.append(episode_obs)
        infos.append(episode_info)


if __name__ == "__main__":
    main()