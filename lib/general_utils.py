import cv2
import os
import numpy as np
from termcolor import colored
if __name__ == '__main__':
    from config_utils import cfg
    from point_cloud_utils import render_point_cloud
else:
    from .config_utils import cfg
    from .point_cloud_utils import render_point_cloud


def _random_demo(
        folder_path: str,
        camera_num: int,
        annot: list,
        round_per_scene: int=10,
        cleaned: bool=False
) -> None:
    for j in cfg.scene_range:
        for i in range(round_per_scene):
            cam = np.random.randint(0, camera_num)
            img_path = annot['ims'][j]['ims'][cam]
            img_path = os.path.join(folder_path, img_path)
            img_og = cv2.imread(img_path)
            if cleaned:
                ply_str = f'result/mesh_cleaned/cleaned_{j:06d}.ply'
            else:
                ply_str = f'result/mesh_transform/transform_{j:06d}.ply'
            img = render_point_cloud(annot['cams']['K'][cam],
                                        annot['cams']['R'][cam],
                                        annot['cams']['T'][cam],
                                        ply_str,
                                        img_og.shape[0], img_og.shape[1],
                                        radius=2)
            display = np.concatenate([img_og, img], axis=1)
            display = cv2.putText(display, f'scene {j} cam {cam}', [5, 25], 
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=1,
                                color=[255,255,255])
            if not os.path.exists('result/demo'):
                os.mkdir('result/demo')
            cv2.imwrite(f'result/demo/scene_{j:04d}_cam_{i:04d}.jpg', display)
    print(colored('please check \'result/demo/\' for demo results', 'green'))
    return
