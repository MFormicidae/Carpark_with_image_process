import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from data_acquire.data import ReadData
from trackpack.counting import Prediction
import winsound

# Create an instance of ReadData
read_data = ReadData()

# Call the get_image_list and get_data methods
image_list, image_path = read_data.get_image_list()
data = read_data.get_data()
start = 1
end = 50
threshold_factors = list(range(start, end+1)) 

filter_configs = [
    # {"filter_name": "bilateral", "noise_filtering": "Gaussian"},
    # {"filter_name": "median", "noise_filtering": "Gaussian"},
    # {"filter_name": "sharp", "noise_filtering": "Gaussian"},
    {"filter_name": "non", "noise_filtering": "Gaussian"},
    # {"filter_name": "bilateral", "noise_filtering": "nonGaussian"},
    # {"filter_name": "median", "noise_filtering": "nonGaussian"},
    # {"filter_name": "sharp", "noise_filtering": "nonGaussian"},
    #{"filter_name": "non", "noise_filtering": "nonGaussian"}
    # {"filter_name": "bilateral", "noise_filtering": "blur"},
    # {"filter_name": "median", "noise_filtering": "blur"},
    # {"filter_name": "sharp", "noise_filtering": "blur"},
    #{"filter_name": "non", "noise_filtering": "blur"}
]

k_sizes = [9]

for config in filter_configs:
    filter_name = config["filter_name"]
    noise_filtering = config["noise_filtering"]
    for ksize in k_sizes:
        for factor in threshold_factors:
            threshold = factor / 100.0  # Convert factor to threshold value
            predict = Prediction(data, image_list, image_path, threshold_factor=threshold, brightness_adjust=0, filter_name=filter_name, noise_filtering=noise_filtering, ksize=ksize)
            predict.process_image(save_runtime=1, consider_all_regions=True, debug=0)
            print(f"Successfully processed with threshold factor: {threshold}, filter: {filter_name}, ksize: {ksize}")
            print("--------------------------------------------------------------------------------------------------------------------")

        winsound.Beep(2000, 500)  
for i in range(3):
    winsound.Beep(2000, 500)  
print("All iterations completed.")
