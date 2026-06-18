# app/image_utils.py
import cv2
import numpy as np

def prep_image_array(img_array, mode="RGB"):
    """
    Takes an ALREADY LOADED image array, flattens it, and adds features.
    """
    h, w, c = img_array.shape
    
    # Flatten the image
    pixel_colors = img_array.reshape((-1, 3))
    pixel_colors = np.float32(pixel_colors)
    
    if mode == "RGB":
        return pixel_colors, (h, w, c)
        
    elif mode == "RGB + XY":
        y_coords, x_coords = np.indices((h, w))
        y_flat = y_coords.flatten()
        x_flat = x_coords.flatten()
        
        x_scaled = (x_flat / w) * 255.0
        y_scaled = (y_flat / h) * 255.0
        
        augmented_pixels = np.column_stack((pixel_colors, x_scaled, y_scaled))
        return np.float32(augmented_pixels), (h, w, c)