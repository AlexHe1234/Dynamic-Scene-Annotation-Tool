# Dynamic Scene Annotation Tool
## Introduction
This is a series of tools combined in one to accelerate dataset processing in dynamic 3d vision tasks. Under the assumption that a dynamic scene is captured by multiple cameras at fixed position over a series of frame time. Given a set of images from different frames, this program will output their corresponding ext, ixt, image/mask paths (combined in certain annotation format), pointcloud (automatically cleaned), and display a simple render result to provide sanity check.
## What to Expect
1. Automatically creates annotations including image paths, camera matrices (both intrinsic and extrinsic).
2. Automatically calls `colmap` for per-scene reconstruction and match coordinates between all scenes.
3. Prunes resulting point cloud according to masks.
## Before Using
To be able to use this program, you need to have [colmap](https://github.com/colmap/colmap) correctly installed and within your `PATH`, you also have to make sure the images (and masks if there are any) are placed in the following way:
1. all images (and masks) must be
.jpg or .png format, and images must have 3 channels (uint8) while masks have 1 channel (uint8).
2. images must be in one of the following arrangements (the naming must be an integer of same digits, but does not require to be 4 or 5, as in this example):
    ```
    a.main folder:
        - images
            - 0000 (this is the index of the camera)
                - 00000.jpg (this is the index of the scene/time)
                - 00001.jpg
                ...
            - 0001
                ...
            ...
            
        - masks (optional)
            ... (corresponding mask must have the exact same name and format as image)
    
    b.main folder: 
        - images
            - 0000 (this is the index of the scene/time)
                - 00000.jpg (this is the index of the camera)
                - 00001.jpg
                ...
            - 0001
                ...
            ...

        - masks (optional)
            ...
    ```
    the first one is referred to as `cs` (camera-scene)
    while the second one as `sc` (scene-camera),
    please specify this (with default being `cs` and `False` for `sc`) in the config file.
3. The resulting annotation file `annot.py` is in the following format:
    ```
    annot.item(): dict
        'ims': list
            scene0: dict
                'ims': list
                    img0: str
                    img1: str
                    ...
            scene1: dict
                'ims': list
                    img0: str
                    img1: str
                    ...
            ...
        'cams': dict
            'K': list
                cam0: numpy.ndarray
                cam1: numpy.ndarray
                ...
            'D': list
                cam0: numpy.ndarray
                cam1: numpy.ndarray
                ...
            'R': list
                cam0: numpy.ndarray
                cam1: numpy.ndarray
                ...
            'T': list
                cam0: numpy.ndarray
                cam1: numpy.ndarray
                ...
    ```
## How to Use
1. Change the configuration settings in `config.py` for specific task.
2. Run `run.py` without any additional arguments.
3. The progress will be displayed in terminal. It also outputs where the results are stored.
## Credits
This tool uses [colmap](https://github.com/colmap/colmap) to reconstruct camera parameters as well as pointclouds.