import numpy as np
import matplotlib.pyplot as plt


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
    for i in range(4):
        _ = a.readline()
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
        exts[j, :3, :3] = quad2rot(quads)
        exts[j, :3, 3] = trans
        a.readline()
    return exts
