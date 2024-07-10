import os
import numpy as np

def generate_depth_map(filename):
    global depth
    ddad_depth = np.load(filename, allow_pickle=True)
    # print(f"ddad_filename: {filename}")
    # print(f"{filename}_depth: {ddad_depth['depth']}")
    try:
        depth = ddad_depth['depth']
    except Exception as e:
        print(f"catch error path: {filename}")
    return depth


if __name__ == '__main__':
    file_count = 0
    file_empty_count = 0
    root_path = '/home/ylc/datasets/DDAD/ddad_train_val'

    for root_path_index in os.listdir(root_path):
        if not root_path_index.isdigit() or int(root_path_index) > 200 :
            continue
        lidar_path = os.path.join(root_path, root_path_index, "depth/lidar/")
        if not os.path.exists(lidar_path):
            print(f'\nfileNotExistPath: {lidar_path}')
            raise FileNotFoundError

        for camera_index in os.listdir(lidar_path):
            camera_path = os.path.join(lidar_path, camera_index)

            for label_map in os.listdir(camera_path):
                label_path = os.path.join(camera_path, label_map)
                if os.path.getsize(label_path) > 0:
                    ddad_data = generate_depth_map(label_path)
                    file_count += 1
                    print(f'scan files: {file_count}', end='\r')
                else:
                    file_empty_count += 1
                    print(f'\nempty files: {file_empty_count}')
                    print(f'empty file path: {label_path}')

    print(f'\n--------Total file number {file_count + file_empty_count}---------')
    print(f'corrent files: {file_count}')
    print(f'empty files: {file_empty_count}')
    print('scan completed')
