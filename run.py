import os
from termcolor import colored
import pycolmap
import shutil
from config import cfg
import logging
import numpy as np
import cv2
from tqdm import tqdm
import datetime
from lib.point_cloud_utils import (
    clean_point_cloud,
    transform_point_cloud,
    get_point_transform,
)
from lib.camera_utils import (
    colmap_cameras_to_ixts,
    colmap_images_to_exts,
    colmap_images_to_exts_unstrict,
)
from lib.time_utils import Est
from lib.general_utils import _random_demo


def main():

    folder_path = cfg.folder
    cs = cfg.cs
    debug = cfg.render_only
    mat_func = cfg.mat_func
    clean_pts = cfg.clean_pts

    assert len(cfg.scene_range) > 0, 'there must be as least one scene'

    if not os.path.exists('log'):
        os.mkdir('log')

    logging.basicConfig(filename=f'log/{datetime.datetime.now().strftime("%m %d %Y %H:%M:%S")}.log', 
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
        for i in tqdm(cfg.scene_range):  # for each scene
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
    d = []
    D = np.zeros((5, 1))

    if not debug and not annot_only and (cfg.scene_range[0] == 0):
        if os.path.exists('colmap'):
            shutil.rmtree('colmap')
            logging.info('removed colmap database folder')
        os.makedirs('colmap')

    est = Est(len(cfg.scene_range))  # create time estimator
    if not debug and not annot_only:
        logging.info('recontructing scenes')
    # colmap reconstruct
    for i in cfg.scene_range:  # for each scene
        logging.info(f'starting scene {i:06d}')
        est.start()
        output_path = 'colmap' + f'/{i:06d}'
        image_dir = os.path.join('tmp', f'{i:06d}')
        if not debug and not annot_only and (i in cfg.scene_range):
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path)
        # mvs_path = output_path + '/mvs'
        database_path = output_path + '/database.db'
        count = 0
        # invalid for annot_only
        while count < cfg.fail_max:
            count += 1
            if not debug and not annot_only:
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
                    if i == cfg.scene_range[0]:
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

        if i == cfg.scene_range[0]:
            if not annot_only:
                ext0 = exts
            for j in range(camera_count):
                if not annot_only:
                    r.append(exts[j, :3, :3])
                    t.append(exts[j, :3, 3:])
                    k.append(ixts[j])
                    d.append(D)
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
                    d.append(D)
            annot['cams'] = {'K': k, 'R': r, 'T': t, 'D': d}
            np.save('result/annots.npy', annot)
            print(colored('annotation has been saved to "result/annots.npy"', 'green'))
            logging.info('\tannotation has been saved')
        else:  # move point cloud to match coordinate in first frame
            if not annot_only:
                rn, tn = get_point_transform(ext0[index], exts)
                transform_point_cloud(f'result/mesh_raw/raw_{i:06d}.ply', rn, tn)

        est.stop()
        logging.info('\ttime left: ' + est.est())

    if not annot_only:
        print(colored(f'meshes have been saved to "result/"', 'green'))
        logging.info('all meshes are saved')
    if clean_pts and not annot_only:
        if use_mask:
            logging.info('cleaning pointclouds')
            clean_point_cloud(folder_path, annot)
            logging.info('pointclouds cleaning complete')
        else:
            raise ValueError('no masks found for cleaning point clouds')
    if not annot_only:
        _random_demo(folder_path, camera_count, annot, cleaned=clean_pts)
    logging.info('all complete')


if __name__ == '__main__':
    main()

