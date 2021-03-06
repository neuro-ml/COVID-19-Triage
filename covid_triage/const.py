# preprocessing hyperparameters
PIXEL_SPACING = (2, 2)  # images are rescaled to the fixed pixel spacing
WINDOW = (-1000, 300)  # image intensities are clipped to the fixed window and rescaled to (0, 1)

# postprocessing hyperparameters
LUNGS_SGM_THRESHOLD = .5
COVID_SGM_THRESHOLD = .2
