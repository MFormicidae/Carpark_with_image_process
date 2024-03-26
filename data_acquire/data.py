import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class ReadData:
    def __init__(self):
        self.labels_pos = r'D:\Carpark_with_image_process\benchmark_Dataset\labels\labels_all62_vgg.json'
        self.image_path = r"D:\Carpark_with_image_process\benchmark_Dataset\Images"

    def get_data(self):
        with open(self.labels_pos) as f:
            data = json.load(f)
        return data

    def get_image_list(self):
        files = os.listdir(self.image_path)
        return files, self.image_path

if __name__ == "__main__":
    read_data = ReadData()
    data = read_data.get_data()
    files =  read_data.get_image_list()

    # Iterate over each image in the data
    for filename, image_data in data.items():
        # Extract the image filename and annotations
        filename = os.path.join(read_data.image_path, filename)
    
        annotations = image_data['regions']
        # print(image_data)
        # Create the figure and plot the image
        fig, ax = plt.subplots()
        image = plt.imread(filename)
        ax.imshow(image)

        # Create and plot the polygons for each annotation
        for annotation in annotations.values():
            shape = annotation['shape_attributes']
            label = annotation['region_attributes']['label']
            points = [(x, y) for x, y in zip(shape['all_points_x'], shape['all_points_y'])]
            polygon = Polygon(points, closed=True, fill=False, color='red', linewidth=2)
            ax.add_patch(polygon)

            # Add label text for better visualization
            x_center = sum(points[i][0] for i in range(len(points))) / len(points)
            y_center = sum(points[i][1] for i in range(len(points))) / len(points)
            ax.text(x_center, y_center, label, ha='center', va='center', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))

        # Adjust plot settings and display
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Image Annotations')
        plt.show()

