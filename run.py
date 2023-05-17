import os
from termcolor import colored
import pycolmap
import shutil
from config import cfg
import logging
import time
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm


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
        if not (0 <= pixel[1] < width and 0 <= pixel[0] < height):
            continue
        img = cv2.circle(img, [int(pixel[0]), int(pixel[1])], radius, color=color[i % 7], thickness=2*radius)
    return img.astype(np.uint8)


def colmap_cameras_to_ixts(
    cameras_path: str,
    camera_num: int
) -> np.ndarray:
    ixts = np.zeros((camera_num, 3, 3))
    file = open(cameras_path, 'r')
    for i in range(3):
        _ = file.readline()
    for i in range(camera_num):
        line = file.readline().split()
        idx = int(line[0]) - 1
        f = float(line[4])
        w = float(line[5])
        h = float(line[6])
        ixts[idx, 0, 0] = ixts[idx, 1, 1] = f
        ixts[idx, 0, 2] = w
        ixts[idx, 1, 2] = h
        ixts[idx, 2, 2] = 1.
    return ixts


def quad2rot(quad: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quad[0], quad[1], quad[2], quad[3]
    r = np.zeros((3, 3))
    r[0, 0] = 1-2*(q2**2)-2*(q3**2)
    r[0, 1] = 2*q1*q2-2*q0*q3
    r[0, 2] = 2*(q1*q3+q0*q2)
    r[1, 0] = 2*(q1*q2+q0*q3)
    r[1, 1] = 1-2*(q1**2)-2*(q3**2)
    r[1, 2] = 2*(q2*q3-q0*q1)
    r[2, 0] = 2*(q1*q3-q0*q2)
    r[2, 1] = 2*(q2*q3+q0*q1)
    r[2, 2] = 1-2*(q1**2)-2*(q2**2)
    return r


def colmap_images_to_exts(
    images_path: str,
    image_num: int
) -> np.ndarray:
    """given images.txt from colmap
    extract extrinsic matices as [N, 3, 4] ndarray
    images that are fed into colmap have to be
    named in purely numerical string like 0087.jpg or 34.jpg

    Args:
        images_path (str): path to the images.txt file that colmap outputs
        image_num (int): number of the total images

    Returns:
        np.ndarray: extrinsic matrices in [N, 3, 4]
    """
    
    a = open(images_path)
    exts = np.zeros((image_num, 3, 4))
    quads = np.zeros(4)
    trans = np.zeros(3)
    for _ in range(4):
        _ = a.readline()
    for _ in range(image_num):
        line1 = a.readline().split()
        quads[0] = float(line1[1])
        quads[1] = float(line1[2])
        quads[2] = float(line1[3])
        quads[3] = float(line1[4])
        trans[0] = float(line1[5])
        trans[1] = float(line1[6])
        trans[2] = float(line1[7])
        j = int(line1[9].split('/')[-1][:-4])
        # print(line1[0], f'{i + 1}')
        # assert line1[0] == f'{i + 1}', 'The colmap file format is not support or corrupted'
        exts[j, :3, :3] = quad2rot(quads)
        exts[j, :3, 3] = trans
        a.readline()
    return exts


def colmap_images_to_exts_unstrict(
    images_path: str
) -> np.ndarray:

    a = open(images_path)
    quads = np.zeros(4)
    trans = np.zeros(3)
    for _ in range(3):
        _ = a.readline()
    image_num = int(a.readline().split()[4][:-1])
    exts = np.zeros((image_num, 3, 4))
    index = []
    for i in range(image_num):
        line1 = a.readline().split()
        quads[0] = float(line1[1])
        quads[1] = float(line1[2])
        quads[2] = float(line1[3])
        quads[3] = float(line1[4])
        trans[0] = float(line1[5])
        trans[1] = float(line1[6])
        trans[2] = float(line1[7])
        j = int(line1[9].split('/')[-1][:-4])
        # print(line1[0], f'{i + 1}')
        # assert line1[0] == f'{i + 1}', 'The colmap file format is not support or corrupted'
        exts[i, :3, :3] = quad2rot(quads)
        exts[i, :3, 3] = trans
        index.append(j)
        a.readline()
    return exts, index


class Est:
    def __init__(self, total):
        self.total = total
        self.ing = False
        self.timing = []
    
    def start(self):
        if self.ing:
            raise Exception('already timing')
        self.ing = True
        self.begin = time.time()

    def stop(self):
        if not self.ing:
            raise Exception('timer hasn\'t started')
        self.ing = False
        self.lastest_stop = time.time()
        self.timing.append(self.lastest_stop - self.begin)

    def est(self) -> str:
        count = len(self.timing)
        if count == 0:
            return 'inf'
        avg = 0
        for i in self.timing:
            avg += i
        avg /= count
        est = avg * (self.total - count) - time.time() + self.lastest_stop
        if est < 60:
            return f'{est:.2f} seconds'
        elif est < 3600:
            return f'{int(est / 60)} minutes and {int((est % 60))} seconds'
        elif est < 86400:
            return f'{int(est / 3600)} hours and {int((est % 3600) / 60)} minutes'
        else:
            return f'{int(est / 86400)} days'


def transform_point_cloud(file, r, t):
    pc = o3d.io.read_point_cloud(file)
    pc = np.asarray(pc.points)
    pc_new = pc @ r.T + t
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc_new)
    o3d.io.write_point_cloud(file.replace('raw', 'transform'), point_cloud)
    return


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

            if cfg.strict_center:
                if msk[:, [0, W - 1]].any() or msk[[0, H - 1], :].any():
                    raise ValueError('the object must be in the center of every image, ' + \
                                    f'error found in scene {i} cam {j}')

        pts = o3d.io.read_point_cloud(f'result/mesh_transform/transform_{i:06d}.ply')
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
        if not os.path.exists('result/mesh_cleaned'):
            os.mkdir('result/mesh_cleaned')
        o3d.io.write_point_cloud(f'result/mesh_cleaned/cleaned_{i:06d}.ply', point_cloud)
    return


def _random_demo(
        folder_path: str,
        camera_num: int,
        scene_num: int,
        annot: list,
        round: int=10,
        cleaned: bool=False
) -> None:
    for i in range(round):
        cam = np.random.randint(0, camera_num)
        scene = np.random.randint(0, scene_num)
        img_path = annot['ims'][scene]['ims'][cam]
        img_path = os.path.join(folder_path, img_path)
        img_og = cv2.imread(img_path)
        if cleaned:
            ply_str = f'result/mesh_cleaned/cleaned_{scene:06d}.ply'
        else:
            ply_str = f'result/mesh_transform/transform_{scene:06d}.ply'
        img = render_point_cloud(annot['cams']['K'][cam],
                                     annot['cams']['R'][cam],
                                     annot['cams']['T'][cam],
                                     ply_str,
                                     img_og.shape[0], img_og.shape[1],
                                     radius=2)
        display = np.concatenate([img_og, img], axis=1)
        display = cv2.putText(display, f'scene {scene} cam {cam}', [5, 25], 
                              fontFace=cv2.FONT_HERSHEY_COMPLEX,
                              fontScale=1,
                              color=[255,255,255])
        if not os.path.exists('result/demo'):
            os.mkdir('result/demo')
        cv2.imwrite(f'result/demo/{i:04d}.jpg', display)
    return


def main():

    folder_path = cfg.folder
    cs = cfg.cs
    debug = cfg.render_only
    mat_func = cfg.mat_func
    clean_pts = cfg.clean_pts

    if not os.path.exists('log'):
        os.mkdir('log')
    logging.basicConfig(filename=f'log/{time.asctime()}.log', 
                        format='%(asctime)s - %(message)s',
                        level=logging.INFO)
    logging.info('start')

    if len(mat_func) > 0:
        annot_only = True
        print(colored('using custom camera function', 'yellow'))
        logging.info('using custom camera function')
    else:
        annot_only = False

    if annot_only and debug:
        raise ValueError('can not set render-only True while specifying mat-func')

    assert os.path.exists(os.path.join(folder_path, 'images')), 'no images folder found'

    if os.path.exists(os.path.join(folder_path, 'masks')):
        use_mask = True
        print(colored('found mask', 'yellow'))
        logging.info('using mask')
    else:
        use_mask = False
        print(colored('proceeding without mask', 'yellow'))
        logging.info('mask not found, will not be used')

    # search for camera & scene number
    annot = {}
    # generate image path annot
    ims = []  # list of lists of paths, first scene then camera
    # ! mask will not be added to annotation, instead replace path name when in use
    # ! however to reconstruct with colmap we still need to load mask
    logging.info('creating image path annotation')
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
    logging.info('image path annotation complete')

    annot['ims'] = ims
    print(colored(f'found {camera_count} cameras, {img_cnt} scenes', 'yellow'))
    logging.info(f'found {camera_count} cameras, {img_cnt} scenes')

    if not debug and not annot_only and not cfg.skip_copy:

        # create colmap image folder
        if os.path.exists('tmp'):
            shutil.rmtree('tmp')
            logging.info('removed tmp folder')
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

    if not debug and not annot_only and (cfg.begin_scene == 0):
        if os.path.exists('colmap'):
            shutil.rmtree('colmap')
            logging.info('removed colmap database folder')
        os.makedirs('colmap')

    est = Est(img_cnt)  # create time estimator

    # colmap reconstruct
    for i in range(img_cnt):  # for each scene
        logging.info(f'starting scene {i:06d}')
        est.start()
        output_path = 'colmap' + f'/{i:06d}'
        image_dir = os.path.join('tmp', f'{i:06d}')
        if not debug and not annot_only and i >= cfg.begin_scene:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
        # mvs_path = output_path + '/mvs'
        database_path = output_path + '/database.db'
        count = 0
        while count < cfg.fail_max:
            count += 1
            if not debug and not annot_only and i >= cfg.begin_scene:
                logging.info(f'\tstarting recontruction')
                pycolmap.extract_features(database_path, image_dir)
                pycolmap.match_exhaustive(database_path)
                maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
                maps[0].write(output_path)
                print(colored(f'reconstruction complete for scene {i}', 'green'))
                logging.info(f'\treconstruction complete')
            # load reconstruction
            try:
                if not annot_only:
                    logging.info(f'\tstarting camera extraction')
                    reconstruction = pycolmap.Reconstruction(output_path)
                    print(reconstruction.summary())
                    if not os.path.exists('result'):
                        os.makedirs('result')
                    if not os.path.exists('result/mesh_raw'):
                        os.mkdir('result/mesh_raw')
                    if not os.path.exists('result/mesh_transform'):
                        os.mkdir('result/mesh_transform')
                    reconstruction.export_PLY('result' + f'/mesh_raw/raw_{i:06d}.ply')
                    reconstruction.export_PLY('result' + f'/mesh_transform/transform_{i:06d}.ply')
                    reconstruction.write_text(output_path)
                    # post proc
                    if i == 0:
                        exts = colmap_images_to_exts(output_path + '/images.txt', camera_count)
                        ixts = colmap_cameras_to_ixts(output_path + '/cameras.txt', camera_count)
                    else:
                        exts, index = colmap_images_to_exts_unstrict(output_path + '/images.txt')
                break
            except:
                print(colored('camera extraction failed, restarting', 'red'))
                logging.info('\tcamera extraction failed, restarting')
                if not debug:
                    shutil.rmtree(output_path)
                    os.makedirs(output_path)
                continue

        if count == cfg.fail_max:
            logging.info('\tmaximum fail reached')
            raise RuntimeError('too many failed construction')
        
        print(colored('camera paramters extraction success', 'green'))
        logging.info('\tcamera extraction complete')

        if i == 0:
            ext0 = exts
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
            logging.info('\tannotation has been saved')
        else:  # move point cloud to match coordinate in first frame
            rn, tn = get_point_transform(ext0[index], exts)
            transform_point_cloud(f'result/mesh_raw/raw_{i:06d}.ply', rn, tn)

        est.stop()
        logging.info('\ttime left: ' + est.est())

    if not annot_only:
        print(colored(f'meshes have been saved to "result/"', 'green'))
        logging.info('all meshes are saved')
    if clean_pts:
        if use_mask:
            logging.info('cleaning pointclouds')
            clean_point_cloud(folder_path, annot)
            logging.info('pointclouds cleaning complete')
        else:
            raise ValueError('setting clean_pts to True without masks')
    if not annot_only:
        _random_demo(folder_path, camera_count, img_cnt, annot, cleaned=clean_pts)
    logging.info('all complete')


if __name__ == '__main__':
    main()

