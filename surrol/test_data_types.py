#!/usr/bin/env python3

"""
测试 pybullet_utils 中的函数是否返回正确的数据类型
"""

import numpy as np
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_numpy_array_methods():
    """测试 numpy 数组的方法"""
    print("=== 测试 numpy 数组的方法 ===")
    
    # 测试普通的 numpy 数组
    arr = np.array([1, 2, 3])
    print(f"numpy array: {arr}")
    print(f"type: {type(arr)}")
    print(f"has .tolist(): {hasattr(arr, 'tolist')}")
    print(f".tolist() result: {arr.tolist()}")
    print(f".tolist() type: {type(arr.tolist())}")
    print()
    
    # 测试 PyBullet 返回的元组
    tuple_data = (1.0, 2.0, 3.0)
    print(f"tuple data: {tuple_data}")
    print(f"type: {type(tuple_data)}")
    print(f"has .tolist(): {hasattr(tuple_data, 'tolist')}")
    
    # 转换为 numpy 数组
    numpy_from_tuple = np.array(tuple_data)
    print(f"numpy from tuple: {numpy_from_tuple}")
    print(f"type: {type(numpy_from_tuple)}")
    print(f"has .tolist(): {hasattr(numpy_from_tuple, 'tolist')}")
    print(f".tolist() result: {numpy_from_tuple.tolist()}")
    print()
    
    # 测试 PyBullet 返回的列表
    list_data = [1.0, 2.0, 3.0]
    print(f"list data: {list_data}")
    print(f"type: {type(list_data)}")
    print(f"has .tolist(): {hasattr(list_data, 'tolist')}")
    
    # 转换为 numpy 数组
    numpy_from_list = np.array(list_data)
    print(f"numpy from list: {numpy_from_list}")
    print(f"type: {type(numpy_from_list)}")
    print(f"has .tolist(): {hasattr(numpy_from_list, 'tolist')}")
    print(f".tolist() result: {numpy_from_list.tolist()}")
    print()

def test_with_actual_environment():
    """测试实际环境中的数据类型"""
    print("=== 测试实际环境中的数据类型 ===")
    
    try:
        # 导入必要的模块
        from surrol.tasks.needle_pick import NeedlePick
        from surrol.utils.pybullet_utils import get_object_orientation_in_camera, get_all_object_info_in_camera
        
        # 创建环境
        env = NeedlePick(render_mode=None)  # 不渲染GUI
        
        # 获取视图矩阵
        view_matrix = env.psm1.get_cam_view_matrix()
        
        # 测试 get_object_orientation_in_camera
        print("测试 get_object_orientation_in_camera:")
        obj_ori = get_object_orientation_in_camera(env, view_matrix)
        print(f"object orientation: {obj_ori}")
        print(f"type: {type(obj_ori)}")
        print(f"has .tolist(): {hasattr(obj_ori, 'tolist')}")
        if hasattr(obj_ori, 'tolist'):
            print(f".tolist() result: {obj_ori.tolist()}")
            print(f".tolist() type: {type(obj_ori.tolist())}")
        print()
        
        # 测试 get_all_object_info_in_camera
        print("测试 get_all_object_info_in_camera:")
        obj_pos, obj_ori = get_all_object_info_in_camera(env, view_matrix)
        print(f"object position: {obj_pos}")
        print(f"position type: {type(obj_pos)}")
        print(f"object orientation: {obj_ori}")
        print(f"orientation type: {type(obj_ori)}")
        
        print(f"position has .tolist(): {hasattr(obj_pos, 'tolist')}")
        print(f"orientation has .tolist(): {hasattr(obj_ori, 'tolist')}")
        
        if hasattr(obj_pos, 'tolist') and hasattr(obj_ori, 'tolist'):
            print(f"position .tolist(): {obj_pos.tolist()}")
            print(f"orientation .tolist(): {obj_ori.tolist()}")
        
        env.close()
        print("✓ 环境测试成功")
        
    except Exception as e:
        print(f"✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_numpy_array_methods()
    test_with_actual_environment()
