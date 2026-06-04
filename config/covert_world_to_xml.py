import os
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import yaml

def convert_world_to_yaml(world_file_path, output_file_path):
    """
    读取 .world 文件，提取指定模型信息，并转换为 .yaml 文件。
    """
    try:
        tree = ET.parse(world_file_path)
        root = tree.getroot()
        world = root.find('world')

        prims_data = {}
        keywords_to_extract = ['target_box', 'my_custom_box', 'obstacle', 'wall']

        if world is not None:
            for model in world.findall('model'):
                model_name = model.get('name')
                
                if model_name and any(keyword in model_name for keyword in keywords_to_extract):
                    pose_element = model.find('pose')
                    size_element = model.find('.//link/collision/geometry/box/size')

                    if pose_element is not None and size_element is not None:
                        pose_values = list(map(float, pose_element.text.split()))
                        position = pose_values[:3]
                        roll, pitch, yaw = pose_values[3:]

                        rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
                        quaternion_np = rotation.as_quat()
                        quaternion_list = quaternion_np.tolist()
                        orientation = [quaternion_list[3], quaternion_list[0], quaternion_list[1], quaternion_list[2]]

                        size = list(map(float, size_element.text.split()))

                        prims_data[model_name] = {
                            'position': position,
                            'orientation': orientation,
                            'size': size
                        }

        if prims_data:
            output_data = {'prims': prims_data}
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # ==========================================================
                #  *** 主要修改点 ***
                #  移除了 default_flow_style=False 参数，以允许PyYAML
                #  自动为短列表选择流式（inline）格式。
                # ==========================================================
                yaml.dump(output_data, f, sort_keys=False, indent=2)
            print(f"成功生成文件: {output_file_path}")
        else:
            print(f"在 {world_file_path} 中未找到任何符合条件的模型，不生成文件。")

    except ET.ParseError as e:
        print(f"解析文件 {world_file_path} 时出错: {e}")
    except FileNotFoundError:
        print(f"文件未找到: {world_file_path}")
    except Exception as e:
        print(f"处理文件 {world_file_path} 时发生未知错误: {e}")


def process_worlds_in_directory(base_dir):
    """
    处理指定基础目录下的所有 level 文件夹及其中的 .world 文件。
    """
    if not os.path.isdir(base_dir):
        print(f"错误: 目录 '{base_dir}' 不存在。")
        return

    for i in range(1, 5):
        level_dir = os.path.join(base_dir, f'level{i}')
        if os.path.isdir(level_dir):
            print(f"--- 正在处理文件夹: {level_dir} ---")
            for j in range(1, 11):
                world_filename = f'{j}.world'
                world_file_path = os.path.join(level_dir, world_filename)
                
                output_filename = f'{j}.yaml'
                output_file_path = os.path.join(level_dir, output_filename)

                if os.path.exists(world_file_path):
                    convert_world_to_yaml(world_file_path, output_file_path)
                else:
                    print(f"警告: 文件 {world_file_path} 不存在，已跳过。")
        else:
            print(f"警告: 文件夹 {level_dir} 不存在，已跳过。")


if __name__ == '__main__':
    # 设置您的 worlds 文件夹的根路径
    worlds_base_path = os.environ.get('COMPASS_WORLDS_DIR', os.path.expanduser('~/Documents/worlds'))
    process_worlds_in_directory(worlds_base_path)
