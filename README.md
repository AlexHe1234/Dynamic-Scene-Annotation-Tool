# Alex's Toolbox (atb)

## Introduction
This is a series of tools combined in one to accelerate dataset processing in dynamic 3d vision tasks. Given a set of
images from different scenes, this program will output their ext, ixt, image/mask paths (combined in certain annotation format), pointcloud, bbox and display a simple render result to provide sanity check. 

## Before using
To be able to use this program, you have to make sure the images (and masks if there are any) in the following way:

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
    please specify this (with default being `cs` and `False` for `sc`) while calling the program.
    
3. there is no guarantee that colmap will produce the right results, and even if it does, there might be noise in the pointcloud, you can always edit the pointcloud yourself, simply use this program to do sanity check on the camera parameters.
