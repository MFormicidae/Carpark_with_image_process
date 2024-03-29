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
threshold_factors = list(range(start, end+1))  # Threshold factors from 0.01 to 0.20

for factor in threshold_factors:
    threshold = factor / 100.0  # Convert factor to threshold value
    predict = Prediction(data, image_list, image_path, threshold_factor=threshold)
    predict.process_image(filter_name="bilateral",noise_filtering = "Gaussian", brightness_adjust=0, save_runtime=1, consider_all_regions=True, ksize=3,debug=0)
    print(f"Successfully processed with threshold factor: {threshold}")
    print("--------------------------------------------------------------------------------------------------------------------")
print("All iterations completed.")

winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

