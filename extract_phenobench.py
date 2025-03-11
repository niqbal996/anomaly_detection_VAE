import os
import cv2
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def filter_images(image_list, filter_string):
    return [img for img in image_list if filter_string in img]

def extract_and_save_instances(image_folder, rgb_folder, semantic_folder, output_folder, filter_string=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    crop_folder = os.path.join(output_folder, 'crop')
    weed_folder = os.path.join(output_folder, 'weed')
    
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(weed_folder, exist_ok=True)
    
    image_list = os.listdir(image_folder)
    
    if filter_string:
        image_list = filter_images(image_list, filter_string)
    
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        rgb_image_path = os.path.join(rgb_folder, image_name)
        semantic_image_path = os.path.join(semantic_folder, image_name)
        
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.imread(rgb_image_path)
        semantic_image = cv2.imread(semantic_image_path, cv2.IMREAD_UNCHANGED)
        
        if rgb_image is None:
            print(f"RGB image not found for {image_name}")
            continue
        
        if semantic_image is None:
            print(f"Semantic image not found for {image_name}")
            continue
        
        unique_ids = np.unique(image)
        
        for unique_id in unique_ids[1:]:
            instance_mask = (image == unique_id).astype(np.uint8) * 255
            instance_semantic_mask = instance_mask > 0
            
            if np.any(instance_semantic_mask):
                class_id = np.unique(semantic_image[instance_semantic_mask])[0]
                if class_id == 1:
                    class_folder = crop_folder
                elif class_id == 2:
                    # class_folder = weed_folder
                    continue
                # TODO for now ignore the partial crops and weeds. Need to handle them later. 
                elif class_id == 3 or class_id == 4:
                    continue
                else:
                    continue
                
                instance_image_name = f"{os.path.splitext(image_name)[0]}_id{unique_id}.png"
                instance_image_path = os.path.join(class_folder, instance_image_name)

                # Create a mask for the instance
                instance_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=instance_mask)
                
                # Find the bounding box of the instance
                x, y, w, h = cv2.boundingRect(instance_mask)
                area = w * h
                if area < 10000:
                    continue
                # Crop the instance from the RGB image
                cropped_instance = instance_rgb[y:y+h, x:x+w]
                
                # Create a black background image with a constant size of 300x300
                background = np.zeros((300, 300, 3), dtype=np.uint8)
                
                # Calculate the position to place the cropped instance on the black background
                center_x, center_y = 150, 150
                start_x = max(center_x - w // 2, 0)
                start_y = max(center_y - h // 2, 0)
                end_x = start_x + w
                end_y = start_y + h
                
                # Ensure the cropped instance fits within the 300x300 background
                end_x = min(end_x, 300)
                end_y = min(end_y, 300)
                
                # Place the cropped instance on the black background
                instance_mask_cropped = instance_mask[y:y+h, x:x+w]
                background[start_y:end_y, start_x:end_x][instance_mask_cropped[:end_y-start_y, :end_x-start_x] > 0] = cropped_instance[:end_y-start_y, :end_x-start_x][instance_mask_cropped[:end_y-start_y, :end_x-start_x] > 0]
                cv2.imwrite(instance_image_path, background)
            
def show_image(image, unique_ids, image_name):
    # Create a colormap with unique colors for each instance
    cmap = ListedColormap(np.random.rand(len(unique_ids), 3))

    # Create a color image to visualize the instances
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for idx, unique_id in enumerate(unique_ids):
        if unique_id == 0:
            continue  # Skip background
        mask = (image == unique_id)
        color_image[mask] = (np.array(cmap(idx)[:3]) * 255).astype(np.uint8)

    # Display the color image
    plt.imshow(color_image)
    plt.title(f"Instances in {image_name}")
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    image_folder = '/mnt/e/datasets/phenobench/train/plant_instances'
    rgb_folder = '/mnt/e/datasets/phenobench/train/images'
    semantic_folder = '/mnt/e/datasets/phenobench/train/semantics'
    output_folder = 'phenobench_instances'
    filter_string = '05-15_'  # Set to None if no filtering is needed
    
    extract_and_save_instances(image_folder, rgb_folder, semantic_folder, output_folder, filter_string)