import numpy as np



# Example usage
stats_list = [
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (1.75, 1.2690906251431504e-15, 1.7499999999999984, 1.7500000000000016), 
    (0, 0, (0, 0)), (0, 0, (0, 0))]

# Crop the inhomogeneous data
cropped_stats_list = np.array(crop_inhomogeneous_data(stats_list))
print(cropped_stats_list)