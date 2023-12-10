import os

def parse_filename(filename):
    # Splitting the filename to extract the necessary information
    parts = filename.split('_')
    # print(parts)
    frame_number = parts[3]
    x0 =int(parts[5])
    y0 = int(parts[7])
    x1 = int(parts[9])
    y1 = int(parts[11].split('.')[0])  # Removing file extension
    return frame_number, [x0, y0, x1, y1]

def read_data(base_directory):
    data_dict = {}

    # Walking through the directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.jpg'):  # Check if the file is an image
                frame_number, coordinates = parse_filename(file)
                parent_directory = os.path.basename(root)
                # Adding to the dictionary
                coordinates.append(parent_directory)
                if frame_number in data_dict:
                    
                    data_dict[frame_number].append(coordinates)
                else:
                    
                    data_dict[frame_number] =  [coordinates]

    return data_dict

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


output_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\Layout_small_data_balanced_eyeclosed'  # Define the output directory
ensure_directory_exists(output_directory)  # Create the directory if it doesn't exist

base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\train'  # Replace with dataset path
datas = read_data(base_directory)

output_file = os.path.join(output_directory,'train.txt')   # Output file path
with open(output_file, 'w') as f:
    for frame_number, data in datas.items():
        f.write(f'{frame_number}\n')
            
train_val_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\trainval'  # Replace with dataset path
train_val_data = read_data(train_val_base_directory)
output_file = os.path.join(output_directory,'trainval.txt')   # Output file path

with open(output_file, 'w') as f:
    for frame_number, data in train_val_data.items():
        f.write(f'{frame_number}\n')
        
test_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\test'  # Replace with dataset path
test_data = read_data(test_base_directory)
output_file = os.path.join(output_directory,'test.txt')   # Output file path
with open(output_file, 'w') as f:
    for frame_number, data in test_data.items():
        f.write(f'{frame_number}\n')
        
val_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\val'  # Replace with dataset path
val_data = read_data(val_base_directory)
output_file = os.path.join(output_directory,'val.txt')   # Output file path
with open(output_file, 'w') as f:
    for frame_number, data in val_data.items():
        f.write(f'{frame_number}\n')