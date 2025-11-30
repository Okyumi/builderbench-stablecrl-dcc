import numpy as np
    
_CUSTOM_COLORS = [
    np.array([1.0, 0.49, 0.43, 1.0]),   # coral
    np.array([0.0, 0.58, 0.55, 1.0]),   # teal
    np.array([0.93, 0.78, 0.28, 1.0]),  # mustard
    np.array([0.38, 0.31, 0.86, 1.0]),  # indigo
    np.array([0.98, 0.67, 0.78, 1.0]),  # rose
    np.array([0.13, 0.55, 0.13, 1.0]),  # forest green
    np.array([0.85, 0.37, 0.00, 1.0]),  # burnt orange
    np.array([0.29, 0.63, 0.92, 1.0]),  # sky blue
    np.array([0.56, 0.27, 0.68, 1.0]),  # plum
    np.array([0.44, 0.50, 0.56, 1.0]),  # slate gray
]

_MJX_PARAMS = {
    # env_name : [nconmax, njmax]
    
    # creative tasks
    'creative-1-task1': [16, 84],
    'creative-1-task2': [16, 84],
    'sparse-creative-1-task1': [16, 84],
    'sparse-creative-1-task2': [16, 84],

    'creative-2-task1': [16, 128],
    'sparse-creative-2-task1': [16, 128],

    'creative-3-task1': [24, 144],
    'sparse-creative-3-task1': [24, 144],

    'creative-4-task1': [24, 256],
    'sparse-creative-4-task1': [24, 256],

    'creative-5-task1': [24, 400],
    'sparse-creative-5-task1': [24, 400],

    'creative-6-task1': [24, 576],
    'sparse-creative-6-task1': [24, 576],

    'creative-7-task1': [32, 784],
    'sparse-creative-7-task1': [32, 784],

    'creative-8-task1': [36, 1024],
    'sparse-creative-8-task1': [36, 1024],

    # planar tasks
    'planar-3-cube-1': [16, 84],
    'planar-3-cube-2': [16, 128],
    'planar-3-cube-3': [24, 144],
    'planar-3-cube-4': [24, 256],
    'planar-3-cube-5': [24, 576],
    'sparse-planar-3-cube-1': [16, 84],
    'sparse-planar-3-cube-2': [16, 128],
    'sparse-planar-3-cube-3': [24, 144],
    'sparse-planar-3-cube-4': [24, 256],
    'sparse-planar-3-cube-5': [24, 576],
    'planar-position-3-cube-1': [16, 84],
    'planar-position-3-cube-2': [16, 128],
    'planar-position-3-cube-3': [24, 144],
    'planar-position-3-cube-4': [24, 256],
    'planar-position-3-cube-5': [24, 576],
    'sparse-planar-position-3-cube-1': [16, 84],
    'sparse-planar-position-3-cube-2': [16, 128],
    'sparse-planar-position-3-cube-3': [24, 144],
    'sparse-planar-position-3-cube-4': [24, 256],
    'sparse-planar-position-3-cube-5': [24, 576],

    'planar-4-cube-1': [16, 84],
    'planar-4-cube-2': [16, 128],
    'planar-4-cube-3': [24, 144],
    'planar-4-cube-4': [24, 256],
    'planar-4-cube-5': [24, 576],
    'sparse-planar-4-cube-1': [16, 84],
    'sparse-planar-4-cube-2': [16, 128],
    'sparse-planar-4-cube-3': [24, 144],
    'sparse-planar-4-cube-4': [24, 256],
    'sparse-planar-4-cube-5': [24, 576],
    'planar-position-4-cube-1': [16, 84],
    'planar-position-4-cube-2': [16, 128],
    'planar-position-4-cube-3': [24, 144],
    'planar-position-4-cube-4': [24, 256],
    'planar-position-4-cube-5': [24, 576],
    'sparse-planar-position-4-cube-1': [16, 84],
    'sparse-planar-position-4-cube-2': [16, 128],
    'sparse-planar-position-4-cube-3': [24, 144],
    'sparse-planar-position-4-cube-4': [24, 256],
    'sparse-planar-position-4-cube-5': [24, 576],

}

_TIME_STEPS = {
    # env_name : [sim_dt, ctrl_dt]

    # creative tasks
    'creative-1-task1': [0.005, 0.02],
    'creative-1-task2': [0.005, 0.02],
    'sparse-creative-1-task1': [0.005, 0.02],
    'sparse-creative-1-task2': [0.005, 0.02],

    'creative-2-task1': [0.005, 0.02],
    'sparse-creative-2-task1': [0.005, 0.02],

    'creative-3-task1': [0.005, 0.02],
    'sparse-creative-3-task1': [0.005, 0.02],

    'creative-4-task1': [0.005, 0.02],
    'sparse-creative-4-task1': [0.005, 0.02],

    'creative-5-task1': [0.005, 0.02],
    'sparse-creative-5-task1': [0.005, 0.02],

    'creative-6-task1': [0.005, 0.02],
    'sparse-creative-6-task1': [0.005, 0.02],

    'creative-7-task1': [0.005, 0.02],
    'sparse-creative-7-task1': [0.005, 0.02],

    'creative-8-task1': [0.005, 0.02],
    'sparse-creative-8-task1': [0.005, 0.02],

    # planar tasks
    'planar-3-cube-1': [0.005, 0.02],
    'planar-3-cube-2': [0.005, 0.02],
    'planar-3-cube-3': [0.005, 0.02],
    'planar-3-cube-4': [0.005, 0.02],
    'planar-3-cube-5': [0.005, 0.02],
    'sparse-planar-3-cube-1': [0.005, 0.02],
    'sparse-planar-3-cube-2': [0.005, 0.02],
    'sparse-planar-3-cube-3': [0.005, 0.02],
    'sparse-planar-3-cube-4': [0.005, 0.02],
    'sparse-planar-3-cube-5': [0.005, 0.02],
    'planar-position-3-cube-1': [0.005, 0.02],
    'planar-position-3-cube-2': [0.005, 0.02],
    'planar-position-3-cube-3': [0.005, 0.02],
    'planar-position-3-cube-4': [0.005, 0.02],
    'planar-position-3-cube-5': [0.005, 0.02],
    'sparse-planar-position-3-cube-1': [0.005, 0.02],
    'sparse-planar-position-3-cube-2': [0.005, 0.02],
    'sparse-planar-position-3-cube-3': [0.005, 0.02],
    'sparse-planar-position-3-cube-4': [0.005, 0.02],
    'sparse-planar-position-3-cube-5': [0.005, 0.02],

    'planar-4-cube-1': [0.005, 0.02],
    'planar-4-cube-2': [0.005, 0.02],
    'planar-4-cube-3': [0.005, 0.02],
    'planar-4-cube-4': [0.005, 0.02],
    'planar-4-cube-5': [0.005, 0.02],
    'sparse-planar-4-cube-1': [0.005, 0.02],
    'sparse-planar-4-cube-2': [0.005, 0.02],
    'sparse-planar-4-cube-3': [0.005, 0.02],
    'sparse-planar-4-cube-4': [0.005, 0.02],
    'sparse-planar-4-cube-5': [0.005, 0.02],
    'planar-position-4-cube-1': [0.005, 0.02],
    'planar-position-4-cube-2': [0.005, 0.02],
    'planar-position-4-cube-3': [0.005, 0.02],
    'planar-position-4-cube-4': [0.005, 0.02],
    'planar-position-4-cube-5': [0.005, 0.02],
    'sparse-planar-position-4-cube-1': [0.005, 0.02],
    'sparse-planar-position-4-cube-2': [0.005, 0.02],
    'sparse-planar-position-4-cube-3': [0.005, 0.02],
    'sparse-planar-position-4-cube-4': [0.005, 0.02],
    'sparse-planar-position-4-cube-5': [0.005, 0.02],
}