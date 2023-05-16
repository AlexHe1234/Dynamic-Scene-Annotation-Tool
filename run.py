from lib.general_import import *
import lib.render_point_cloud as rpc
from lib.colmap_txt_to_camera import colmap_images_to_exts
from lib.colmap_txt_to_camera import colmap_cameras_to_ixts
import os
from termcolor import colored
import pycolmap
import shutil
from config import cfg
import sys


def clip(value: any, lower: any, upper: any) -> any:
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value
    
# Disable
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def clean_point_cloud(
        folder_path: str,
        annot: list,
        padding: int=1  # unit in px
) -> None:
    ims_annot = annot['ims']
    cam_annot = annot['cams']
    # check center
    scene_count = len(ims_annot)
    assert scene_count > 0
    cam_count = len(ims_annot[0]['ims'])
    assert cam_count > 0

    print(colored('cleaning point cloud...', 'green'))
    for i in range(scene_count):
        print(f'scene {i}:')
        # make sure this scene is calculatable
        for j in range(cam_count):
            img_path = os.path.join(folder_path, ims_annot[i]['ims'][j])
            msk = cv2.imread(img_path.replace('images', 'masks'), cv2.IMREAD_GRAYSCALE)
            H, W = msk.shape[0], msk.shape[1]
            if msk[:, [0, W - 1]].any() or msk[[0, H - 1], :].any():
                raise ValueError('the object must be in the center of every image')

        pts = o3d.io.read_point_cloud(f'result/mesh_raw_{i:06d}.ply')
        pts = np.asarray(pts.points)

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

            for p in range(pixels.shape[0]):
                pixel = [int(pixels[p][0]), int(pixels[p][1])]
                if not (0 <= pixel[1] < W and 0 <= pixel[0] < H):
                    erase.append(p)

                h_from = clip(pixel[1] - padding, 0, H - 1)
                h_to = clip(pixel[1] + padding + 1, 1, H)
                w_from = clip(pixel[0] - padding, 0, W - 1)
                w_to = clip(pixel[0] + padding + 1, 1, W)

                if not msk[h_from:h_to, w_from:w_to].any():
                    erase.append(p)

            keep = [idx for idx in range(pixels.shape[0]) if idx not in erase]
            pts = pts[keep]

        point_cloud = o3d.geometry.PointCloud()
        # add points
        point_cloud.points = o3d.utility.Vector3dVector(pts)
        # write point cloud
        o3d.io.write_point_cloud(f'result/mesh_cleaned_{i:06d}.ply', point_cloud)
    return


def _random_demo(
        folder_path: str,
        camera_num: int,
        scene_num: int,
        annot: list,
        round: int=10,
        delay: int=1000,
        cleaned: bool=False
) -> None:
    for _ in range(round):
        cam = np.random.randint(0, camera_num)
        scene = np.random.randint(0, scene_num)
        img_path = annot['ims'][scene]['ims'][cam]
        img_path = os.path.join(folder_path, img_path)
        img_og = cv2.imread(img_path)
        if cleaned:
            ply_str = f'result/mesh_cleaned_{scene:06d}.ply'
        else:
            ply_str = f'result/mesh_raw_{scene:06d}.ply'
        img = rpc.render_point_cloud(annot['cams']['K'][cam],
                                     annot['cams']['R'][cam],
                                     annot['cams']['T'][cam],
                                     ply_str,
                                     img_og.shape[0], img_og.shape[1],
                                     radius=2)
        display = np.concatenate([img_og, img], axis=1)
        cv2.imshow('sanity check', display)
        cv2.waitKey(delay)
    return


def main():

    folder_path = cfg.folder
    cs = cfg.cs
    debug = cfg.render_only
    mat_func = cfg.mat_func
    clean_pts = cfg.clean_pts

    if len(mat_func) > 0:
        annot_only = True
        print(colored('using custom camera function', 'yellow'))
    else:
        annot_only = False

    if annot_only and debug:
        raise ValueError('can not set render-only True while specifying mat-func')

    assert os.path.exists(os.path.join(folder_path, 'images')), 'no images folder found'

    if os.path.exists(os.path.join(folder_path, 'masks')):
        use_mask = True
        print(colored('found mask', 'yellow'))
    else:
        use_mask = False
        print(colored('proceeding without mask', 'yellow'))

    # search for camera & scene number
    annot = {}
    # generate image path annot
    ims = []  # list of lists of paths, first scene then camera
    # ! mask will not be added to annotation, instead replace path name when in use
    # ! however to reconstruct with colmap we still need to load mask
    if cs:
        dir_list = [name for name in os.listdir(os.path.join(folder_path, 'images')) if '.' not in name]
        dir_list = sorted(dir_list)
        camera_count = len(dir_list)
        img_cnt = 0
        for i in range(camera_count):
            img_list = [img_file for img_file in os.listdir(os.path.join(folder_path, 'images', dir_list[i])) \
                        if (img_file[-4:] == '.jpg') or (img_file[-4:] == '.png')]
            img_list = sorted(img_list)
            if i == 0:
                img_cnt = len(img_list)
                for img in img_list:
                    ims.append({'ims': [os.path.join('images', dir_list[i], img)]})
            else:
                assert len(img_list) == img_cnt, \
                    f'all cameras must have same amount of scenes, error found in folder {dir_list[i]}'
                for j in range(img_cnt):
                    ims[j]['ims'].append(os.path.join('images', dir_list[i], img_list[j]))
    else:  # sc
        dir_list = [name for name in os.listdir(os.path.join(folder_path, 'images')) if '.' not in name]
        dir_list = sorted(dir_list)
        img_cnt = len(dir_list)
        camera_count = 0
        for i in range(img_cnt):  # for each scene
            img_list = [img_file for img_file in os.listdir(os.path.join(folder_path, 'images', dir_list[i])) \
                        if (img_file[-4:] == '.jpg') or (img_file[-4:] == '.png')]
            img_list = sorted(img_list)
            if i == 0:
                camera_count = len(img_list)
            else:
                assert len(img_list) == camera_count, \
                f'all scenes must have same amount of scenes, error found in folder {dir_list[i]}'
            ims.append({'ims': []})
            for j in range(camera_count):  # for each camera
                ims[i]['ims'].append(os.path.join('images', dir_list[i], img_list[j]))

    annot['ims'] = ims
    print(colored(f'found {camera_count} cameras, {img_cnt} scenes', 'yellow'))

    if not debug and not annot_only and not cfg.skip_copy:

        # create colmap image folder
        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
        os.makedirs('tmp')

        print(colored('copying scenes', 'yellow'))
        for i in tqdm(range(img_cnt)):  # for each scene
            os.makedirs(f'tmp/{i:06d}')
            for j in range(camera_count):  # for each camera
                # os.symlink(os.path.join(folder_path, ims[i][j]), f'tmp/{i:06d}/{j:06d}.jpg')
                img = cv2.imread(folder_path + '/' + ims[i]['ims'][j])
                if use_mask:
                    msk = cv2.imread((folder_path + '/' + ims[i]['ims'][j]).replace('images', 'masks'), cv2.IMREAD_GRAYSCALE)
                    img[msk == 0] = 0
                cv2.imwrite(f'tmp/{i:06d}/{j:06d}.jpg', img.astype(np.uint8))

    # first scene then camera
    k = []  # scene * camera * [3, 3]
    r = []  # scene * camera * [3, 3]
    t = []  # scene * camera * [3, 1]
    D = np.zeros((5, 1))

    if not debug and not annot_only:
        if os.path.exists('colmap'):
            shutil.rmtree('colmap')
        os.makedirs('colmap')

    # colmap reconstruct
    for i in range(img_cnt):  # for each scene

        output_path = 'colmap' + f'/{i:06d}'
        image_dir = os.path.join('tmp', f'{i:06d}')
        if not debug and not annot_only:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
        # mvs_path = output_path + '/mvs'
        database_path = output_path + '/database.db'
        if not debug and not annot_only:
            pycolmap.extract_features(database_path, image_dir)
            pycolmap.match_exhaustive(database_path)
            maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
            maps[0].write(output_path)
        print(colored(f'reconstruction complete for scene {i}', 'green'))
        # load reconstruction
        if not annot_only:
            reconstruction = pycolmap.Reconstruction(output_path)
            print(reconstruction.summary())
            if not os.path.exists('result'):
                os.makedirs('result')
            reconstruction.export_PLY('result' + f'/mesh_raw_{i:06d}.ply')
            reconstruction.write_text(output_path)
            # post proc
            exts = colmap_images_to_exts(output_path + '/images.txt', camera_count)
            ixts = colmap_cameras_to_ixts(output_path + '/cameras.txt', camera_count)

        if i != 0:
            continue
        
        for j in range(camera_count):
            if not annot_only:
                r.append(exts[j, :3, :3])
                t.append(exts[j, :3, 3:])
                k.append(ixts[j])
            else:
                if mat_func[-3:] == '.py':
                    mat_func = mat_func[:-3]
                try:
                    ret_mat = __import__(mat_func).ret_mat
                except:
                    raise ImportError('ret_mat function is not found')
                krt = ret_mat(j)
                assert isinstance(krt, np.ndarray), 'return type of ret_mat must be numpy.ndarray'
                assert krt.shape == (3, 7), 'shape of ret_mat return must be [3, 7]'
                r.append(krt[:3, 3:6])
                t.append(krt[:3, 6:])
                k.append(krt[:3, :3])
        annot['cams'] = {'K': k, 'R': r, 'T': t, 'D': D}
        np.save('result/annot.npy', annot)
        print(colored('annotation has been saved to "result/annot.npy"', 'green'))
    if not annot_only:
        print(colored(f'meshes have been saved to "result/"', 'green'))
    if clean_pts:
        if use_mask:
            clean_point_cloud(folder_path, annot)
        else:
            raise ValueError('setting clean_pts to True without masks')
    if not annot_only:
        _random_demo(folder_path, camera_count, img_cnt, annot, cleaned=clean_pts)


if __name__ == '__main__':
    main()

