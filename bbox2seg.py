import cv2
import numpy as np
import os
import sys
import torch
from pathlib import Path
from ultralytics import SAM
import yaml

def yolo_to_pixel_coordinates(yolo_values, image_width, image_height):
    _, center_x, center_y, width, height = yolo_values
    center_x_pixel = int(center_x * image_width)
    center_y_pixel = int(center_y * image_height)
    width_pixel = int(width * image_width)
    height_pixel = int(height * image_height)
    top_left_x = center_x_pixel - (width_pixel // 2)
    top_left_y = center_y_pixel - (height_pixel // 2)
    bottom_right_x = center_x_pixel + (width_pixel // 2)
    bottom_right_y = center_y_pixel + (height_pixel // 2)
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

def read_yolov4_labels(label_file_path, image_width, image_height):
    labels = []
    class_ids = []
    with open(label_file_path, 'r') as file:
        for line in file:
            yolo_values = [float(val) for val in line.strip().split()]
            pixel_coords = yolo_to_pixel_coordinates(yolo_values, image_width, image_height)
            labels.append(pixel_coords)
            class_id = int(line.strip().split()[0])
            class_ids.append(class_id)
    return class_ids, torch.tensor(labels)

def get_image_size(image_path):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    return image, image_width, image_height

def convert_to_results(image_path, label_path):
    print(f"Pre-processing {image_file}")
    image, image_width, image_height = get_image_size(image_path)
    class_ids, label_tensor = read_yolov4_labels(label_path, image_width, image_height)
    result = {'path': image_path, 'boxes': label_tensor, 'orig_img': image, 'classes': class_ids}
    return result

def sam_annotate(result, sam_model, label_output_dir, device=''):
    class_ids = result['classes']
    if len(class_ids):
        print(f"SAM is now processing {result['path']}")
        sam_results = sam_model(result['orig_img'], bboxes=result['boxes'], verbose=False, save=False, device=device)
        segments = sam_results[0].masks.xyn
        outfile = str(Path(label_output_dir) / Path(result['path']).stem) + '.txt'
        with open(outfile, 'w') as f:
            for i in range(len(segments)):
                s = segments[i]
                if len(s) == 0:
                    continue
                segment = map(str, segments[i].reshape(-1).tolist())
                f.write(f'{class_ids[i]} ' + ' '.join(segment) + '\n')
        return outfile
    
def generate_colors(n):  
    colors = {}  
    for i in range(n):  
        angle = 2 * np.pi * i / n  
        r = int(255 * max(0, np.cos(angle)))  
        g = int(255 * max(0, np.cos(angle + 2 * np.pi / 3)))  
        b = int(255 * max(0, np.cos(angle + 4 * np.pi / 3)))  
        colors[i] = (r, g, b)  
    return colors  

def visualise_segment(seg_label_path, seg_img_dir, image_path, image_file, num_classes):
    colors = generate_colors(num_classes)
    img = cv2.imread(image_path)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    with open(seg_label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            class_id = int(data[0])
            polygon = np.array([[int(float(data[i+1])*img.shape[1]), img.shape[0]-int(float(data[i])*img.shape[0])] for i in range(1, len(data)-1, 2)], np.int32)
            img = cv2.polylines(img, [polygon], True, colors[int(class_id)], 2)
    print(f"Visualising segment for {image_file}")
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(seg_img_dir, image_file), img)

def get_num_classes(yaml_file):  
    with open(yaml_file, 'r') as stream:  
        yaml_content = yaml.safe_load(stream)  
        return yaml_content['nc']  

if __name__ == '__main__':
    try: 
        data_yaml = sys.argv[1]
        image_dir = sys.argv[2]
        label_dir = sys.argv[3]
        sanity = sys.argv[4]
        is_large = sys.argv[5]
    except IndexError:
        print("Usage: python bbox2seg.py YAML_PATH IMAGE_DIRECTORY LABEL_DIRECTORY SEG_IMAGES(y/n) USE_LARGE(y/n)\nPlease input required data.")
        data_yaml = input("Enter path to data.yaml for dataset: ")
        image_dir = input("Enter path to directory containing images: ")
        label_dir = input("Enter path to directory containg yolo-format labels: ")
        sanity = input("Output a copy of all images with segmentation masks applied? (y/n): ")
        is_large = input("Use SAM (Large) instead of SAM (Base)? (y/n): ")
    try:
        if sanity == 'y':
            print("Output of images with masked applied is enabled.")
            seg_img_dir = Path(str(image_dir)).parent / 'seg_images'
            Path(seg_img_dir).mkdir(exist_ok=True, parents=True)
            print(f"Saving output images to {seg_img_dir}")
        label_out_dir = Path(str(image_dir)).parent / 'seg_labels'
        Path(label_out_dir).mkdir(exist_ok=True, parents=True)
        print(f"Saving output labels to {label_out_dir}")
        if is_large == 'y':
            sam_model = SAM('sam_l.pt')
            print("Using SAM (Large) model.")
        else:
            sam_model = SAM('sam_b.pt')
            print("Using SAM (Base) model.")
        for image_file, label_file in zip(os.listdir(image_dir), os.listdir(label_dir)):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            result = convert_to_results(image_path, label_path)
            num_classes = get_num_classes(data_yaml)
            seg_label = sam_annotate(result, sam_model, label_out_dir)
            if sanity == 'y':
                visualise_segment(seg_label, seg_img_dir, image_path, image_file, num_classes)
    except Exception as e:
        print(f"{e}\nAborting segmentation")
        sys.exit()
