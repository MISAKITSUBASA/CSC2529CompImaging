import xml.etree.ElementTree as ET
import argparse
import cv2
import face_detection
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

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

# Example usage
base_directory = 'D:\\DSFD-Pytorch-Inference-1\\data\\actors'  # Replace with your dataset path
data = read_data(base_directory)

print(data)

import os
import xml.etree.ElementTree as ET

def create_xml_file(frame_number, objects, output_folder):
    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder")
    folder.text = "VOC2007"

    filename = ET.SubElement(annotation, "filename")
    print(frame_number)
    filename.text = f"{frame_number}.jpg"

    # Add other static information here (source, owner, etc.)
    # ...

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = "1280"  # Replace with actual width if known
    ET.SubElement(size, "height").text = "720"  # Replace with actual height if known
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    # Create an object element for each object
    for obj in objects:
        object_elem = ET.SubElement(annotation, "object")
        ET.SubElement(object_elem, "name").text = obj[4]  # Name from your data
        # Add other elements like pose, truncated, difficult if needed
        ET.SubElement(object_elem, "difficult").text = "1"
        bndbox = ET.SubElement(object_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj[0])
        ET.SubElement(bndbox, "ymin").text = str(obj[1])
        ET.SubElement(bndbox, "xmax").text = str(obj[2])
        ET.SubElement(bndbox, "ymax").text = str(obj[3])
        

    # Create a new XML file
    tree = ET.ElementTree(annotation)
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    with open(f"{output_folder}/{frame_number}.xml", "wb") as file:
        tree.write(file)
        
output_folder = "xml_data"  

for frame, values in data.items():
    create_xml_file(frame, values, output_folder)
