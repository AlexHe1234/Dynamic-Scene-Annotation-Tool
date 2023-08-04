# Dynamic Scene Annotation Tool
## Introduction
```
images & masks -> intrinsics & extrinsics & pointclouds 
```
This is a series of tools combined in one to accelerate dataset processing in dynamic 3d vision tasks using `PyCOLMAP`. Under the assumption that a dynamic scene is captured by multiple cameras at fixed positions over a series of time frames. 

Given a set of images from different frames, this program will output their corresponding exts, ixts, images/masks paths (stored in single .npy file), pointclouds (automatically cleaned), and display a simple render result to provide sanity check.

In summary, this tool:
1. Creates annotations including image paths, camera matrices (both intrinsic and extrinsic matrices).
2. Calls `COLMAP` for per-scene reconstruction and match coordinates between all scenes.
3. Prunes resulting point cloud according to masks.
## Get Started
### 1. Install PyCOLMAP
To be able to use this program, you need to have [COLMAP](https://github.com/colmap/colmap) correctly installed and within your `PATH`. Compile with cuda support might provide better speed (although I was having trouble seeing that, perhaps they haven't support cuda yet on certain tasks).
### 2. Arrange Data Files
you also have to make sure the images (and masks if there are any) are placed in the following way:
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
### 3. Sit back and Start Reconstructing
1. Change the configuration settings in `config.py` for specific task.
2. Run `run.py` without any additional arguments.
3. The progress will be displayed in terminal and recorded in detail in log files (under log folder). It also outputs where the results are stored.
### 4. Result
All results will be under 'result' folder.
1. The resulting annotation file `annots.npy` can be loaded using
    ```
    annots = np.load('path\to\annots.npy', allow_pickle=True)
- and will be in the following format:

    ```
    annots.item(): dict
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
2. The resulting point clouds (meshes without faces, so why not call them meshes) are divded into `mesh_cleaned` (cleaned using masks), `mesh_raw` (raw colmap output files) and `mesh_transform` (transformed to match the camera annotations since raw outputs are not aligned)
3. The testing results to see the results are under `demo` folder, I find this helpful since sometimes colmap doesn't give the correct result and this provides an easy way to tell that.
## Miscellaneous
1. If you find demo results to be incorrect, simply replace `scene_range` in config file and have another go. I find that sufficient to generate plausible results for most of the scenes.
## Credits
This tool uses [colmap](https://github.com/colmap/colmap) to reconstruct camera parameters as well as pointclouds.



