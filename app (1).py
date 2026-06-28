import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load disease model
disease_model = tf.keras.models.load_model('plant_disease_model_final.keras', compile=False)

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_healthy'
]

def analyze_image_properties(img_array):
    """
    Analyzes multiple visual properties to determine if image is a leaf.
    Real leaves have specific color distributions, textures, and patterns.
    """
    r = img_array[:,:,0].astype(float)
    g = img_array[:,:,1].astype(float)
    b = img_array[:,:,2].astype(float)
    
    # Convert to HSV-like analysis
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    saturation = np.where(max_channel > 0, (max_channel - min_channel) / max_channel, 0)
    
    # Leaf characteristics:
    # 1. Has reasonable saturation (not pure black/white/grey)
    mean_saturation = saturation.mean()
    
    # 2. Green or yellow-brown dominance (healthy or diseased leaves)
    green_ratio = g / (r + g + b + 1e-6)
    red_ratio = r / (r + g + b + 1e-6)
    blue_ratio = b / (r + g + b + 1e-6)
    
    mean_green = green_ratio.mean()
    mean_red = red_ratio.mean()
    mean_blue = blue_ratio.mean()
    
    # 3. Check for artificial/digital image signatures
    # Digital artwork tends to have very uniform color regions
    # Real photos have natural color variance
    color_variance = np.std(img_array.astype(float))
    
    # 4. Check for very dark backgrounds (common in studio leaf photos - OK)
    # vs pure black backgrounds (common in digital art - suspicious)
    dark_pixel_ratio = np.mean(max_channel < 30)
    
    # 5. Natural image entropy - real photos have higher entropy
    hist_r = np.histogram(r, bins=32)[0]
    hist_g = np.histogram(g, bins=32)[0]
    hist_b = np.histogram(b, bins=32)[0]
    
    def entropy(hist):
        hist = hist[hist > 0] / hist.sum()
        return -np.sum(hist * np.log2(hist))
    
    avg_entropy = (entropy(hist_r) + entropy(hist_g) + entropy(hist_b)) / 3
    
    return {
        'mean_saturation': mean_saturation,
        'mean_green': mean_green,
        'mean_red': mean_red,
        'mean_blue': mean_blue,
        'color_variance': color_variance,
        'dark_pixel_ratio': dark_pixel_ratio,
        'avg_entropy': avg_entropy
    }

def is_valid_leaf(img):
    """
    Multi-factor leaf validation system.
    Returns (is_leaf: bool, reason: str)
    """
    img_array = np.array(img.resize((224, 224)))
    props = analyze_image_properties(img_array)
    
    # Rule 1: Reject pure digital/dark images
    # Digital art like Kali logo has >60% dark pixels + low entropy
    if props['dark_pixel_ratio'] > 0.5 and props['avg_entropy'] < 3.0:
        return False, "Image appears to be digital artwork or logo"
    
    # Rule 2: Reject images dominated by blue channel
    # Real leaves are never predominantly blue
    if props['mean_blue'] > props['mean_green'] and props['mean_blue'] > props['mean_red']:
        if props['mean_blue'] > 0.4:
            return False, "Image color profile does not match any crop leaf"
    
    # Rule 3: Reject very low saturation images (greyscale/screenshots)
    if props['mean_saturation'] < 0.08:
        return False, "Image appears to be greyscale or has insufficient color"
    
    # Rule 4: Reject very low entropy (solid colors, logos, simple graphics)
    if props['avg_entropy'] < 2.5:
        return False, "Image appears to be a graphic or logo"
    
    # Rule 5: Accept if reasonable green presence
    # (even diseased brown/yellow leaves retain some green)
    if props['mean_green'] > 0.28:
        return True, "Valid leaf detected"
    
    # Rule 6: Accept brownish/yellowish leaves (diseased)
    # Brown leaves: red+green dominant, low blue
    is_brownish = (props['mean_red'] > 0.3 and 
                   props['mean_green'] > 0.25 and 
                   props['mean_blue'] < 0.25)
    if is_brownish:
        return True, "Valid leaf detected"
    
    # Rule 7: Accept if high natural variance (real photo characteristic)
    if props['color_variance'] > 45 and props['avg_entropy'] > 4.0:
        return True, "Valid image detected"
    
    return False, "Image does not appear to be a crop leaf"

def predict_disease(img):
    if img is None:
        return {"Please upload an image": 1.0}
    
    img_rgb = img.convert('RGB')
    img_array_full = np.array(img_rgb)
    
    # Validate if image is a leaf
    valid, reason = is_valid_leaf(img_rgb)
    
    if not valid:
        return {"Not a recognized crop leaf - Please upload a Tomato, Potato or Pepper leaf image": 1.0}
    
    # Run disease detection
    img_resized = img_rgb.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = disease_model.predict(img_array, verbose=0)[0]
    max_confidence = float(np.max(predictions))

    if max_confidence < 0.4:
        return {"Low confidence - Please upload a clearer leaf image": 1.0}

    results = {}
    for i, class_name in enumerate(class_names):
        clean_name = class_name.replace('___', ' - ').replace('_', ' ')
        results[clean_name] = float(predictions[i])

    return results

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Disease Prediction"),
    title="Crop Disease Detection AI",
    description="Upload a Tomato, Potato, or Pepper leaf image to detect disease. Built with ResNet50 transfer learning - 92.6% validation accuracy.",
    examples=None,
    flagging_mode="never"
)

demo.launch()