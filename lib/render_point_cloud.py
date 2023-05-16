"""render a pointcloud given:
k: [3, 3] intrinsic matrix
r & t: [3, 3] and (3,)/[3, 1]/[1, 3] extrinsic matrix
pointcloud: path to a ply file or [N, 3] npy file or a ndarray of shape [N, 3]
height & width: for the desired image
radius: radius of one dot in resulting image, unit in px"""


from .general_import import *


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
