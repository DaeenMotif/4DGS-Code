# Explain these default kplane config parameters
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4], # # scales only the spatial dimensions
    # to encourage spatial smoothness
    
    # tiny MLP depth
    defor_depth = 1, # depth (number of hidden layers) of the deform MLP at encoder
    net_width = 128, # number of neurons per layer of the deform MLP
    plane_tv_weight = 0.0002, # spatial planes regularization weight of total-variational loss (tv)
    # encourages spatially smooth features in the hexplane grids
    
    time_smoothness_weight = 0.001,  # for spatial-temporal planes
    # encourages temporally smooth deformations
    
    l1_time_planes =  0.0001, # weight for the L1 penalty pushing spatiotemporal planes toward 1.0
    render_process=True
)
OptimizationParams = dict(
    # dataloader=True,
    iterations = 14_000,
    batch_size=2,
    coarse_iterations = 3000, # initial warm-up with basic 3DGS
    densify_until_iter = 10_000,
    opacity_reset_interval = 300000, # not used
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)
