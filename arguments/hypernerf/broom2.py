_base_="default.py"
# Kplanes config
# Kplanes factorizes a higher dimensional space, d, into collection of 2D (K-D) planes
# C(d,2): convert the 4D neural voxels representation to C(4,2) = 6 2D feature grids (spatial grids and spatio-temporal grids)
ModelParams=dict(
    kplanes_config = {
     'grid_dimensions': 2, # K = 2
     'input_coordinate_dim': 4, # d = 4 input coord dim (x,y,z,t)
     'output_coordinate_dim': 16, # feature channel, each of the 6 planes' per grid sample stores a feature of 16 dim
     'resolution': [64, 64, 64, 100] # discretizes the normalized 3D BBox [-1,1] into 64x64x64 base voxels
     # discretizes the normalized time range into 100 temporal bins
    },
)
OptimizationParams=dict(
)
