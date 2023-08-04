from yacs.config import CfgNode as CN


cfg = CN()


"""main options"""

# main folder directory that includes images/ (and optionally masks/)
cfg.folder = '/home/idarc/hgz/datasets/blender/lego_dynamic'

# select the list of scenes to reconstruct
cfg.scene_range = [0, 1, 2]

# default cs, if sc set False
# cs means scene images are under camera folders
# sc means camera images are under scene folders
cfg.cs = True


"""custom options"""

# clean pointcloud according to the masks
# when setting this to True,
# masks folder must be represent
# this does not contradict with reconstruction
cfg.clean_pts = True

# generate annotation and render sanity check without reconstruction,
# based on previously reconstructed results,
# set this to true when reconstruction HAS been run
cfg.render_only = True

# the name of the .py file that contains the function
# must be in the same directory as "run.py"
# "ret_mat(cam: >= 0 int) -> 3*7 [k, r, t]",
# reconstruction will not be called if this is not None
cfg.mat_func = ''

# skip copy only if the files in tmp
# folder belongs to the current task
cfg.skip_copy = False

# maximum fail try
cfg.fail_max = 5
