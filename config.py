from yacs.config import CfgNode as CN


cfg = CN()

# main folder directory that includes images/ (and optionally masks/)
cfg.folder = '/home/idarc/hgz/datasets/blender/lego_dynamic'

# default cs, if sc set False
cfg.cs = True

# clean pointcloud according to the masks
# when setting this to True,
# masks folder must be represent
# this does not contradict with reconstruction
cfg.clean_pts = True

# generate annotation and render sanity check without reconstruction,
# based on previously reconstructed results,
# set this to true when reconstruction HAS been run
cfg.render_only = False

# the path to the .py file that contains the function
# "ret_mat(cam: >= 0 int) -> 3*7 [k, r, t]",
# reconstruction will not be called if this is not None
cfg.mat_func = ''

# skip copy only if the files in tmp
# folder belongs to the current task
cfg.skip_copy = True

# select the scene to begin reconstruction with
cfg.begin_scene = 0

# strict centering, might 
# accidentally over-prune point cloud
# if set this to False
cfg.strict_center = False