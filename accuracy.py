import re
#special_slots
#normal_slots
folder_name = "run"
range_file = 2 
accumulate_accuracy_value = []
# Function to read accuracy value from file
def read_accuracy_value(file_path):
    accuracy_value = None
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(accuracy_pattern, content)
        if match:
            accuracy_value = float(match.group(1))
    return accuracy_value


for i in range(range_file):
    file_path = f'./{folder_name}/run{i+1}/threshold_accuracy.txt'

    # Regular expression pattern to extract accuracy value
    accuracy_pattern = r'accuracy_value: ([\d.]+)'
    # Read accuracy value from file
    accuracy_value = read_accuracy_value(file_path)

    if accuracy_value is not None:
        
        accumulate_accuracy_value.append(accuracy_value)
    else:
        print("Accuracy value not found in the file.")

print(f"{folder_name} = {accumulate_accuracy_value}")