import json
import sys

# Assuming your JSON data is stored in a file named "data.json"
with open("D:\\carparking_count 0.1\\dataset\\labels\\labels_my-project-name_2024-02-29-05-39-20.json", "r") as f:
    data = json.load(f)

def generate_json_object(image_id, id_num, segment_values, bbox_values, area_value):
    return f'''{{
        "id": {id_num},
        "iscrowd": 0,
        "image_id": {image_id},
        "category_id": 2,
        "segmentation": {segment_values},
        "bbox": [
            {bbox_values[0]},
            {bbox_values[1]},
            {bbox_values[2]},
            {bbox_values[3]}
        ],
        "area": {area_value}
    }}'''

# Write header to the file
with open("head.txt", "w") as header_file:
    for i in range(46, 105):
        header_file.write(f'''{{
            "id": {i},
            "width": 1280,
            "height": 720,
            "file_name": "{i}.jpg"
        }},\n''')

# Write JSON objects to the output file
with open("output.txt", "w") as output_file:
    for i in range(46, 105):
        for j in range(16):
            output_file.write(generate_json_object(i, j, data["annotations"][j]["segmentation"], data["annotations"][j]["bbox"], data["annotations"][j]["area"]) + ',\n')
