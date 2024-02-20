import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from track_position.data import ReadData
from trackpack.counting import Prediction

# Create an instance of ReadData
read_data = ReadData()

# Call the get_image_list and get_data methods
image_list, image_path = read_data.get_image_list()
data = read_data.get_data()


predict = Prediction(0.2)
predict.process_image(image_list, data, image_path, 1)
  