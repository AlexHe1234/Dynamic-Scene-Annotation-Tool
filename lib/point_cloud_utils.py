import numpy as np
from termcolor import colored
import os
import open3d as o3d
from tqdm import tqdm
import cv2
if __name__ == '__main__':
    from config_utils import cfg
else:
    from .config_utils import cfg


def clip(value: any, lower: any, upper: any) -> any:
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


def clean_point_cloud(
        folder_path: str,
        annot: list,
        padding: int=1  # unit in px
) -> None:

    ims_annot = annot['ims']
    cam_annot = annot['cams']
    # check center
    cam_count = len(ims_annot[0]['ims'])
    assert cam_count > 0
    print(cam_count)
    print(colored('cleaning point cloud...', 'green'))
    for i in cfg.scene_range:
        print(f'scene {i}:')

        pts_ = o3d.io.read_point_cloud(f'result/mesh_transform/transform_{i:06d}.ply')
        pts = np.asarray(pts_.points)
        color = np.asarray(pts_.colors)
        # when the cast of a point is outside the mask, delete it from point cloud
        for j in tqdm(range(cam_count)):
            k = cam_annot['K'][j]
            r = cam_annot['R'][j]
            t = cam_annot['T'][j]
            
            t = t.reshape(3, 1)
            coord_cam = r.dot(pts.T) + t
            coord_img = k.dot(coord_cam)
            coord_img /= coord_img[2, :]
            pixels = coord_img[:2, :].T

            img_path = os.path.join(folder_path, ims_annot[i]['ims'][j])
            msk = cv2.imread(img_path.replace('images', 'masks'), cv2.IMREAD_GRAYSCALE)

            H, W = msk.shape[0], msk.shape[1]

            erase = []

            # ! test view

            for p in range(pixels.shape[0]):
                pixel = [int(pixels[p][0]), int(pixels[p][1])]
                if not (0 <= pixel[0] < W and 0 <= pixel[1] < H):
                    continue

                h_from = clip(pixel[1] - padding, 0, H - 1)
                h_to = clip(pixel[1] + padding + 1, 1, H)
                w_from = clip(pixel[0] - padding, 0, W - 1)
                w_to = clip(pixel[0] + padding + 1, 1, W)

                if not msk[h_from:h_to, w_from:w_to].any():
                    erase.append(p)

            keep = [idx for idx in range(pixels.shape[0]) if idx not in erase]

            pts = pts[keep]
            color = color[keep]

        point_cloud = o3d.geometry.PointCloud()
        # add points
        point_cloud.points = o3d.utility.Vector3dVector(pts)
        point_cloud.colors = o3d.utility.Vector3dVector(color)
        # write point cloud
        if not os.path.exists('result/mesh_cleaned'):
            os.mkdir('result/mesh_cleaned')
        o3d.io.write_point_cloud(os.path.join('result/mesh_cleaned', f'cleaned_{i:06d}.ply'), point_cloud)
    return


def render_point_cloud(
        k: np.ndarray, 
        r: np.ndarray,
        t: np.ndarray,
        pointcloud: str,
        height: int,
        width: int,
        radius: int=1
    ) -> np.ndarray:
    
    color = [[255, 0, 0],
             [0, 255, 255],
             [0, 0, 255],
             [255, 0, 255],
             [255, 255, 255],
             [0, 255, 0],
             [255, 255, 0]]

    if isinstance(pointcloud, str):
        if '.ply' in pointcloud:
            mesh = o3d.io.read_point_cloud(pointcloud)
            pts = np.asarray(mesh.points)
        elif '.npy' in pointcloud:
            pts = np.load(pointcloud)
            if not (len(pts.shape) == 2 and pts.shape[1] == 3):
                raise IndexError
        else:
            raise NotImplementedError
    elif isinstance(pointcloud, np.ndarray):
        assert pointcloud.shape[1] == 3, 'Point cloud needs to be [N, 3] in shape'
        pts = pointcloud
    else:
        raise NotImplementedError

    t = t.reshape(3, 1)
    coord_cam = r.dot(pts.T) + t
    coord_img = k.dot(coord_cam)
    coord_img /= coord_img[2, :]
    pixels = coord_img[:2, :].T
    
    img = np.zeros((height, width, 3))
    
    for i in range(pixels.shape[0]):
        pixel = pixels[i]
        if not (0 <= pixel[1] < height and 0 <= pixel[0] < width):
            continue
        img = cv2.circle(img, [int(pixel[0]), int(pixel[1])], radius, color=color[i % 7], thickness=2*radius)
    return img.astype(np.uint8)


# finding the rotation and translation
# matrix between the point cloud and
# the one in the first frame to
# match coordinate
def get_rot_trans(
        tar: np.ndarray, 
        src: np.ndarray
) -> np.ndarray:
    # they are both N*3 matrices with same N
    center_tar = np.average(tar, axis=0)
    center_src = np.average(src, axis=0)

    r_src = src - center_src
    r_tar = tar - center_tar

    r = np.linalg.lstsq(r_src, r_tar, rcond=None)[0]
    t = tar - src @ r

    return r.T, t[0]


def transform_point_cloud(file, r, t):
    pc_ = o3d.io.read_point_cloud(file)
    pc = np.asarray(pc_.points)
    pc_new = pc @ r.T + t
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_new)
    point_cloud.colors = pc_.colors
    o3d.io.write_point_cloud(file.replace('raw', 'transform'), point_cloud)
    return


def get_point_transform(
        ext0: np.ndarray,  # [N, 3, 4]
        extn: np.ndarray,  # [N, 3, 4]
) -> np.ndarray:
    
    r0 = ext0[:, :3, :3]
    t0 = ext0[:, :3, 3:]
    rn = extn[:, :3, :3]
    tn = extn[:, :3, 3:]

    pts0 = -np.transpose(r0, [0, 2, 1]) @ t0
    ptsn = -np.transpose(rn, [0, 2, 1]) @ tn

    return get_rot_trans(pts0[..., 0], ptsn[..., 0])
