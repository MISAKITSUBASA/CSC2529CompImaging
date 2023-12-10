import os

def parse_filename(filename):
    parts = filename.split('_')
    frame_number = parts[3]
    x0 = int(parts[5])
    y0 = int(parts[7])
    x1 = int(parts[9])
    y1 = int(parts[11].split('.')[0])  # Removing file extension
    return frame_number, [x0, y0, x1, y1]

def read_data(base_directory):
    data_dict = {}
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.jpg'):
                frame_number, coordinates = parse_filename(file)
                parent_directory = os.path.basename(root) # label 
                
                if parent_directory in data_dict:
                    data_dict[parent_directory].append(frame_number)
                else:
                    data_dict[parent_directory] = [frame_number]
    return data_dict

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\train'  # Replace with your dataset path
datas = read_data(base_directory)

output_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\Main_small_data_balanced_eyeclosed'  # Define the output directory
ensure_directory_exists(output_directory)  # Create the directory if it doesn't exist

for name, frames in datas.items():
    output_file = os.path.join(output_directory, f'{name}_train.txt')  # Output file path

    with open(output_file, 'w') as f:
        for fram in frames:
            f.write(f'{fram}\n')
            
            
            
trainval_data_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\trainval'  # Replace with your dataset path
trainval_datas = read_data(trainval_data_base_directory)

for name, frames in trainval_datas.items():
    output_file = os.path.join(output_directory, f'{name}_trainval.txt')  # Output file path

    with open(output_file, 'w') as f:
        for fram in frames:
            f.write(f'{fram}\n')
            
            
            
            
test_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\test'  # Replace with your dataset path
test_data = read_data(test_base_directory)
output_file = 'test.txt'  # Output file path

for name, frames in test_data.items():
    output_file = os.path.join(output_directory, f'{name}_test.txt')  # Output file path

    with open(output_file, 'w') as f:
        for fram in frames:
            f.write(f'{fram}\n')
        
val_base_directory = 'small_data_set_balanced_closed_eye_removed_balanced\\val'  # Replace with your dataset path
val_data = read_data(val_base_directory)
output_file = 'val.txt'  # Output file path

for name, frames in val_data.items():
    output_file = os.path.join(output_directory, f'{name}_val.txt')  # Output file path

    with open(output_file, 'w') as f:
        for fram in frames:
            f.write(f'{fram}\n')