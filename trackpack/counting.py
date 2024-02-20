import cv2
import numpy as np
import os
import matplotlib.pyplot as plt 
from PIL import Image
from collections import defaultdict
class Prediction():
    def __init__(self, threshold_factor) -> None:#
        self.threshold_factor = threshold_factor
        self.total_correct_predictions = 0
        self.total_regions_processed = 0
        self.filename_list = []
        self.accuracy_accumulated =[]
        self.false_images_predict=[]
        self.TP = 0  # True Positives
        self.FP = 0  # False Positives
        self.FN = 0  # False Negatives
        self.TN = 0  # True Negatives
        self.save_run = False
        
        self.region_data = []

    def process_image(self, image_list, data, image_path, save_runtime =None):
        if save_runtime == 1 :
            folder_name="run"
            self.save_run = True
            # Create the parent folder if it doesn't exist
            parent_folder = f"./{folder_name}"
            os.makedirs(parent_folder, exist_ok=True)

            # Find the next available folder number
            num_list = 1
            while os.path.exists(f"{parent_folder}/{folder_name}{num_list}"):
                num_list += 1

            # Create the runtime folder
            self.runtime_path = os.path.join(parent_folder, f"{folder_name}{num_list}")
            os.makedirs(self.runtime_path)
        else:
            pass

        for filename in image_list:
            if filename not in data:
                print(f"Error: {filename} not found in data.")
                return

            annotations = data[filename]["regions"]
            
            if not annotations:
                print(f"No annotations found for {filename}.")
                return

            image = cv2.imread(os.path.join(image_path, filename))
            if image is None:
                print(f"Error: Failed to load image {filename}.")
                return
            
            self.filename_list.append(filename)
            # Preprocess the image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 2)
            imgThreshold = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 11)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            labels_dataset = []
            results_predicted = []
              # Store data for each region
            images = []
            correct_predictions = 0  #count for correct predictions
            total_regions = 0  #  count for total regions
            warning_occurred = False
            vis_image = image.copy()
            bw_image = imgDilate.copy()
            # region_data.append(filename)
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
                if non_count < count_threshold :
                    color = (0, 255, 0) #Green = free
                    labels_predicted = 0
                else : 
                    color = (0, 0, 255) #Red = full
                    labels_predicted = 1 
                label = str(non_count)

                # Draw the polygon on the visualization image
                vis_image = cv2.polylines(vis_image, [points], True, color, thickness=2)

                # Add region number and label text for better visualization
                x_center = int(np.mean(shape['all_points_x']))
                y_center = int(np.mean(shape['all_points_y']))
                #check labels in dataset and labels predicted are equal or not
            
                total_regions += 1
                if data_labels == "Full":  
                    labels_dataset.append(1)
                    if labels_predicted == 1 : #True positive 
                        correct_predictions += 1
                        self.TP += 1
                        self.region_data.append({
                            "region":i,
                            "result": 1
                        }) 
                    
                    else : #error occure is False negative 
                        images.append(vis_image)
                        self.FP += 1
                        warning_occurred = True
                        self.region_data.append({
                            "region":i,
                            "result": 0
                        }) 
                #True positive is predicted result is Ture and Actually label from dataset is ture    
                elif data_labels == "Empty":  
                    labels_dataset.append(0)
                    if labels_predicted == 0: #True negative
                        self.TN += 1
                        correct_predictions += 1
                        self.region_data.append({
                            "region":i,
                            "result": 1
                        }) 
                    else :#error occure is False positive
                        self.FN +=1
                        warning_occurred = True
                        images.append(vis_image)
                        self.region_data.append({
                            "region":i,
                            "result": 0
                        }) 
                cv2.putText(vis_image, f"{i}", (x_center, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,152,255), 2, cv2.LINE_AA)
                cv2.putText(vis_image, label, (x_center, y_center + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
            # Define the dimensions of the grid
            if warning_occurred:
                self.false_image(filename,vis_image)
                self.false_image(f"BW{filename}",bw_image)
                self.false_images_predict.append(vis_image)
            else:
                self.total_correct_predictions =+1
            accuracy = (correct_predictions / total_regions) 
            self.accuracy_accumulated.append(accuracy)
        
        
        if self.save_run == True:
            image_read = self.load_images_from_folder(self.runtime_path,1)
            bw_read = self.load_images_from_folder(self.runtime_path,2)
            # Display or save the resulting mosaic
            self.delete_files(self.runtime_path,1)
            self.delete_files(self.runtime_path,2)
            self.save_image_mosaic(image_read, len(self.false_images_predict)//3, len(self.false_images_predict)//4,"False_Return")
            self.save_image_mosaic(bw_read,len(self.false_images_predict)//3, len(self.false_images_predict)//4,"BW_False_Return")
            self.summary_result()
            self.summary_image()
            self.accumulate_region_accuracy(self.region_data)
        # print()
    def accumulate_region_accuracy(self, data):
                # Create a dictionary to store counts of correct and total predictions for each region
        region_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Iterate through the data and update counts
        for entry in data:
            region = entry['region']
            result = entry['result']
            if result == 1:
                region_counts[region]['correct'] += 1
            region_counts[region]['total'] += 1

        # Calculate accuracy for each region
        accuracy_per_region = {}
        for region, counts in region_counts.items():
            accuracy_per_region[region] = counts['correct'] / counts['total']
        regions = list(accuracy_per_region.keys())
        accuracies = [accuracy_per_region[region] for region in regions]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(regions, accuracies, color='skyblue')
        plt.xlabel('Region')
        plt.ylabel('Accuracy')
        plt.title('Region vs. Accuracy')
        plt.xticks(regions)
        plt.ylim(0, 1)  # Limit y-axis to range between 0 and 1 for accuracy
        plt.grid(axis='y', linestyle='--')
        plt.savefig(os.path.join(self.runtime_path, "Accuracy For Each Region"))


    def delete_files(self, folder_path,select):
        if self.save_run == True:

            if select == 1:
                # List all files in the folder
                files = os.listdir(folder_path)
                # Iterate through each file and delete it
                for file_name in files:
                    if file_name[0].isdigit():
                        file_path = os.path.join(folder_path, file_name)
                        if os.path.isfile(file_path):
                            # print(file_path)
                            os.remove(file_path)
            if select == 2:
                # List all files in the folder
                files = os.listdir(folder_path)
                # Iterate through each file and delete it
                for file_name in files:
                    if file_name.startswith('B'):
                        file_path = os.path.join(folder_path, file_name)
                        if os.path.isfile(file_path):
                            # print(file_path)
                            os.remove(file_path)
    
    # def false_binary_image(self,filename ,image):
    def false_image(self, filename ,image):
        if self.save_run== True:
            cv2.imwrite(os.path.join(self.runtime_path, f"{filename}_processed.jpg"), image)
    
    def load_images_from_folder(self, folder,select):
        if select == 1:
            processed_images = []
            for filename in sorted(os.listdir(folder)):
                if filename[0].isdigit():
                    img = cv2.imread(os.path.join(folder, filename))
                    if img is not None:
                        processed_images.append(img)
            return processed_images
        elif select == 2:
            processed_images = []
            for filename in sorted(os.listdir(folder)):
                if filename.startswith('B'):
                    img = cv2.imread(os.path.join(folder, filename))
                    if img is not None:
                        processed_images.append(img)
            return processed_images

    def save_image_mosaic(self, images, rows, cols, file_name):
        # Calculate the maximum number of images to plot in the mosaic
        max_images = rows * cols
        if len(images) > max_images:
            images = images[:max_images]  # Truncate the list to contain only the first max_images

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
            false_save_time_path = os.path.join(self.runtime_path, f"{file_name}.jpg")
            cv2.imwrite(false_save_time_path, mosaic)
        else:
            cv2.imshow("False_Return", mosaic)

        
    def summary_result(self):
        #create confusion matrix
        confusion_matrix = np.array([[self.TP, self.FP],
                                     [self.FN, self.TN]])
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        classes = ['Full', 'Empty']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(confusion_matrix[i][j]))
        if self.save_run == True: 
            plt.savefig(os.path.join(self.runtime_path, 'confusion_matrix.png'))
     
    def summary_image(self):
        # Plotting bar chart for all images
        plt.figure()#create plot figure 
        plt.bar(self.filename_list , self.accuracy_accumulated,color='skyblue') #create bar graph
        #create a label
        plt.xlabel('Image')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for Each Image')
        plt.xticks(rotation=60)
        plt.grid(axis='y', linestyle='--')  
        plt.tight_layout()
        if self.save_run == True: 
            plt.savefig(os.path.join(self.runtime_path, 'Accuracy for Each Image.png'))

   