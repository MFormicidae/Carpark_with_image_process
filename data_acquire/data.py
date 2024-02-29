import json
import os

class ReadData:
    def __init__(self):
        self.labels_pos = r'D:\carparking_count 0.1\dataset\labels\master_normal_vgg.json'
        self.image_path = r"D:\carparking_count 0.1\dataset\images"

    def get_data(self):
        with open(self.labels_pos) as f:
            data = json.load(f)
            return data
        
    def get_image_list(self):
        files = os.listdir(self.image_path)
        return files , self.image_path
    
if __name__ == "__main__":
    read_data = ReadData()
    data = read_data.get_data()
    files =  read_data.get_image_list()
    print(files[0])
