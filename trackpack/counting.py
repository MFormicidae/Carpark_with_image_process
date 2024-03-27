import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 
from PIL import Image
from collections import defaultdict
import time
import seaborn as sns
from scipy.datasets import face
from scipy.signal import wiener

class Prediction():
    def __init__(self, data=None, image_list=None , image_path=None, threshold_factor =None ) -> None:
         
        self.start_time = time.time()
        print("program running do not have any operation....")
        self.create_runtime_folder()
        self.normal = 44
        self.data = data
        self.threshold_factor = threshold_factor
        self.save_run = False
        self.region_data = []
        self.debug = False
        self.image_data = {}
        for filename in image_list:
            image = cv2.imread(os.path.join(image_path, filename))
            if image is not None:
                self.image_data[filename] = image
            else:
                print(f"Error: Failed to load image {filename}.")
     
    def process_image(self,filter_name=None,noise_filtering = None, brightness_adjust=None, save_runtime=None,consider_all_regions=False, ksize = None, debug=None):
        self.total_correct_predictions = 0
        self.total_regions_processed = 0
        self.filename_list = []
        self.accuracy_accumulated =[]
        self.TP = 0  # True Positives
        self.FP = 0  # False Positives
        self.FN = 0  # False Negatives
        self.TN = 0  # True Negatives
        self.consider_all_regions = consider_all_regions
        self.debug = debug == 1
        self.save_run = save_runtime == 1
        self.filter_name = filter_name

        for filename, image in self.image_data.items():
            self.file_name = filename
            if filename not in self.data:
                print(f"Error: {filename} not found in data.")
                return

            annotations = self.data[filename]["regions"]
            if not annotations:
                print(f"No annotations found for {filename}.")
                return

            self.filename_list.append(filename)
            blurred_image = self.preprocess_image(image, brightness_adjust,noise_filtering,ksize=ksize)
            if self.consider_all_regions :
                annotations = {key: annotations[key] for idx, key in enumerate(annotations.keys()) if idx < self.normal}
            else:
                annotations = annotations
            
            if filter_name in ["median", "bilateral", "sharp"]:
                imgDilate ,filtered_image,image_threshold= self.apply_filter(blurred_image, filter_name)
                processed_image = self.process_annotations(annotations, blurred_image, imgDilate,image)

        false_result = self.load_images_from_folder(1)
        false_bW = self.load_images_from_folder(2)
        true_result = self.load_images_from_folder(3)
        true_bw = self.load_images_from_folder(4)
        self.delete_files()
        self.save_image_mosaic(false_result,3,4,"False_Result")
        self.save_image_mosaic(false_bW,3, 4,"False_BW")
        self.save_image_mosaic(true_result,3, 4,"True_Result")
        self.save_image_mosaic(true_bw,3,4 ,"True_BW")
        self.summary_result()
        self.summary_image()
        self.accumulate_region_accuracy(self.region_data,consider_all_regions= True)
        self.accumulate_region_accuracy(self.region_data)
        self.save_image(f"Pre-process_{filename}",blurred_image) 
        self.save_image(f"Dilate_image{filename}",imgDilate) 
        self.save_image(f"Processed_{filename}",processed_image)
        self.save_image(f"filtered_{filename}",filtered_image)      
        self.save_image(f"Image_threshold{filename}",image_threshold) 
        end_time = time.time()  # Record end time
        elapsed_time = end_time - self.start_time
        print(f"Total time taken for processing: {elapsed_time:.2f} seconds")
        print(f"Success! your image has been processed with {filter_name} filter,your file saved at {self.runtime_path}")
        
        
    def preprocess_image(self, image, brightness_adjust , noise_filtering ,ksize):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if brightness_adjust == 1:
            image = cv2.convertScaleAbs(image, 1.5, 3)
        if noise_filtering == "Gaussian":
            image = cv2.GaussianBlur(image, (ksize,ksize), 3)
        elif noise_filtering == "blur":
            image = cv2.blur(image,(ksize,ksize))
        
        return image    
    
    def apply_filter(self, image, filter_name):
        # Adaptive thresholding
        image_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        
        # Applying specified filter
        if filter_name == "median":
            # Median filtering
            filtered_image = cv2.medianBlur(image_threshold, 5)
        elif filter_name == "bilateral":
            # Bilateral filtering
            filtered_image = cv2.bilateralFilter(image_threshold, 9, 200, 200)
            # Global thresholding after bilateral filtering
            _, filtered_image = cv2.threshold(filtered_image, 126, 255, cv2.THRESH_BINARY)
        elif filter_name == "sharp":  
            kernel3 = np.array([[-1, -1,  -1],
                                [-1,  8, -1],
                                [-1, -1,  -1]])
            filtered_image = cv2.filter2D(src=image_threshold, ddepth=-1, kernel=kernel3)
           
        # Dilation
        kernel = np.ones((3, 3), np.uint8)
        image = cv2.dilate(filtered_image, kernel, iterations=1)
        
        return image, filtered_image, image_threshold

    def create_runtime_folder(self):
        folder_name = "run"
        parent_folder = f"./{folder_name}"
        os.makedirs(parent_folder, exist_ok=True)

        num_list = 1
        while os.path.exists(f"{parent_folder}/{folder_name}{num_list}"):
            num_list += 1

        self.runtime_path = os.path.join(parent_folder, f"{folder_name}{num_list}")
        os.makedirs(self.runtime_path)
        

    def process_annotations(self, annotations, blurred_image, imgDilate,image):
        vis_image = image.copy()
        labels_dataset = []
        total_regions = 0
        correct_predictions = 0
        warning_occurred = False  # Initialize warning_occurred
        for i, annotation in enumerate(annotations.values(), start=1):
            data_labels = annotation.get('region_attributes', {}).get('label', 'No Label')
            shape = annotation['shape_attributes']
            points = np.array(list(zip(shape['all_points_x'], shape['all_points_y'])), np.int32)
            points = points.reshape((-1, 1, 2))
            mask = np.zeros_like(blurred_image)
            cv2.fillPoly(mask, [points], 255)
            cropped_count_image = cv2.bitwise_and(blurred_image , mask)
            cropped_bi_image = cv2.bitwise_and(imgDilate, mask)
            count_crop = cv2.countNonZero(cropped_count_image)
            non_count = cv2.countNonZero(cropped_bi_image)

            # Determine color and label based on threshold
            count_threshold = count_crop * self.threshold_factor
            if non_count < count_threshold:
                color = (0, 255, 0)  # Green = free
                labels_predicted = 0
            else:
                color = (0, 0, 255)  # Red = full
                labels_predicted = 1
            label = str(non_count)
            # Draw the polygon on the visualization image
            vis_image = cv2.polylines(vis_image, [points], True, color, thickness=2)

            # Add region number and label text for better visualization
            x_center = int(np.mean(shape['all_points_x']))
            y_center = int(np.mean(shape['all_points_y']))
            # check labels in dataset and labels predicted are equal or not

            total_regions += 1
            if data_labels == "Full":
                labels_dataset.append(1)
                if labels_predicted == 1:  # True positive
                    correct_predictions += 1
                    self.TP += 1
                    self.region_data.append({
                        "region": i,
                        "result": 1
                    })

                else:  # error occur is False negative                    
                    self.FP += 1
                    warning_occurred = True
                    self.region_data.append({
                        "region": i,
                        "result": 0
                    })
                    if self.debug:
                        print(self.file_name)
                        print(f"Error occurred in region {i}:")
                        print(f"Data labels: {data_labels}, Predicted labels: {labels_predicted}")
                        print(f"Count threshold: {count_threshold}, Total count:{count_crop}")
                        print(f"Counted in region {i}:{non_count}")
                        print("--------------------------------------------------")
            # True positive is predicted result is True and Actually label from dataset is True
            elif data_labels == "Empty":
                labels_dataset.append(0)
                if labels_predicted == 0:  # True negative
                    self.TN += 1
                    correct_predictions += 1
                    self.region_data.append({
                        "region": i,
                        "result": 1
                    })
                else:  # error occur is False positive
                    self.FN += 1
                    warning_occurred = True
                    self.region_data.append({
                        "region": i,
                        "result": 0
                    })
                    if self.debug:
                        print(self.file_name)
                        print(f"Error occurred in region {i}:")
                        print(f"Data labels: {data_labels}, Predicted labels: {labels_predicted}")
                        print(f"Count threshold: {count_threshold}, Total count:{count_crop}")
                        print(f"Counted in region {i}:{non_count}")
                        print("--------------------------------------------------")
            cv2.putText(vis_image, f"{i}", (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 152, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(vis_image, label, (x_center, y_center + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 152, 255), 1, cv2.LINE_AA)

        # Define the dimensions of the grid
        if warning_occurred:
            self.save_image(f"False_Result{self.file_name}", vis_image)
            self.save_image(f"False_BW_{self.file_name}", imgDilate)
    

        self.total_correct_predictions += 1  
        self.save_image(f"True_Result_{self.file_name}", vis_image)
        self.save_image(f"True_BW_{self.file_name}", imgDilate)
        accuracy = (correct_predictions / total_regions)
        self.accuracy_accumulated.append(accuracy)
        return vis_image
    def load_images_from_folder(self, select=None):
        folder = self.runtime_path



        processed_images = []
        if select is not None:
            if select == 1:
                prefix = 'False_Result'
            elif select == 2:
                prefix = 'False_BW'
            elif select == 3:
                prefix = 'True_Result'
            elif select == 4:
                prefix = 'True_BW'
            else:
                raise ValueError("Invalid select value. Select must be between 1 and 4.")

            for filename in sorted(os.listdir(folder)):
                if filename.startswith(prefix):
                    img = cv2.imread(os.path.join(folder, filename))
                    if img is not None:
                        processed_images.append(img)
        else:
            for filename in sorted(os.listdir(folder)):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    processed_images.append(img)

        return processed_images

    def delete_files(self):
        if self.save_run :
            folder_path = self.runtime_path
             # List all files in the folder
            files = os.listdir(folder_path)
            # Iterate through each file and delete it
            for file_name in files:
                if file_name.startswith(file_name):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        # print(file_path)
                        os.remove(file_path)
           
    def summary_image(self, consider_all_regions=False):
        plt.figure()#create plot figure 
        plt.bar(self.filename_list , self.accuracy_accumulated,color='skyblue') #create bar graph
        #create a label
        plt.xlabel('Image')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for Each Image')
        plt.xticks(rotation=85)
        plt.grid(axis='y', linestyle='--')  
        plt.tight_layout()
        if self.save_run == True: 
            plt.savefig(os.path.join(self.runtime_path, 'Accuracy for Each Image.png'))
            
            
    def accumulate_region_accuracy(self, data, consider_all_regions=False):
        # Create a dictionary to store counts of correct and total predictions for each region
        region_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Iterate through the data and update counts
        for entry in data:
            region = entry['region']
            result = entry['result']
            # Filter out regions outside the range 1-16 if not considering all regions
            if consider_all_regions or (1 <= region <= self.normal):
                if result == 1:
                    region_counts[region]['correct'] += 1
                region_counts[region]['total'] += 1

        # Calculate accuracy for each region
        accuracy_per_region = {}
        for region, counts in region_counts.items():
            accuracy_per_region[region] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0

        regions = list(accuracy_per_region.keys())
        accuracies = [accuracy_per_region[region] for region in regions]
        
        # Plotting
        plt.figure()
        plt.bar(regions, accuracies, color='skyblue')
        plt.xlabel('Region')
        plt.ylabel('Accuracy')
        plt.title('Region vs. Accuracy')
        plt.xticks(regions,fontsize=5)
        plt.ylim(0, 1)  # Limit y-axis to range between 0 and 1 for accuracy
        plt.grid(axis='y', linestyle='--')
        
        if self.save_run:
            filename ="Accuracy_For_Normal_Region.png"
            plt.savefig(os.path.join(self.runtime_path, filename))
        else:
            plt.show()
    def save_image_mosaic(self, images, rows, cols, file_name):
        # Calculate the maximum number of images to plot in the mosaic
       
        num_images = len(images)
        # Check if there are no images
        if num_images == 0:
            return
        # Calculate the number of rows and columns based on the number of images
        rows = min(rows, num_images)
        cols = min(cols, num_images)
        while rows * cols < num_images:
            if cols > rows:
                rows += 1
            else:
                cols += 1

        # Calculate the dimensions of the mosaic
        mosaic_height = images[0].shape[0] * rows
        mosaic_width = images[0].shape[1] * cols
        
        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(images):
                    start_x = j * images[idx].shape[1]
                    start_y = i * images[idx].shape[0]
                    end_x = start_x + images[idx].shape[1]
                    end_y = start_y + images[idx].shape[0]
                    mosaic[start_y:end_y, start_x:end_x] = images[idx]
        
        if self.save_run:
            false_save_time_path = os.path.join(self.runtime_path, f"Normal_{file_name}.jpg")
            cv2.imwrite(false_save_time_path, mosaic) 
            
        else:
            cv2.imshow("False_Return", mosaic)
    def summary_result(self):
        # Create confusion matrix
        confusion_matrix = np.array([[self.TP, self.FP],
                                     [self.FN, self.TN]])
        
        all_confusion_matrix = self.TP + self.TN + self.FN + self.FP
        print(self.TP + self.TN,self.FP + self.FN)
        accuracy_matrix = (self.TP + self.TN) / all_confusion_matrix
        print(f"threshold_value: {self.threshold_factor}, accuracy_value: {accuracy_matrix}")

        classes = ['Full', 'Empty']
        # Plot confusion matrix
        plt.figure()
        sns.set(font_scale=1)  
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes, cbar=False)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
       
        if self.save_run:
            with open(f"{self.runtime_path}/threshold_accuracy.txt", "w") as f:
                f.write(f"threshold_value: {self.threshold_factor}, accuracy_value: {accuracy_matrix}, filter_name: {self.filter_name}")
            plt.savefig(os.path.join(self.runtime_path, 'confusion_matrix.png'))

     
    def save_image(self, filename ,image):
        if self.save_run:
            cv2.imwrite(os.path.join(self.runtime_path, filename), image)

