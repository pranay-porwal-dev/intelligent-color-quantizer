# app/app.py
import streamlit as st
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as calculate_ssim

from image_utils import prep_image_array
from ml_engine import run_kmeans, get_wcss_array

st.set_page_config(layout="wide", page_title="Color Quantizer & Visualizer")
st.title("🎨 ML Color Quantization & Cluster Visualization")

# --- Resource Management ---
@st.cache_data
def load_and_resize_image_from_upload(uploaded_file, max_dim=800):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_img(img, max_dim)

@st.cache_data
def load_and_resize_image_from_path(file_path, max_dim=800):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return resize_img(img, max_dim)

def resize_img(img, max_dim):
    h, w, c = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# --- Advanced Metrics Calculation ---
def calculate_metrics(original_img_3d, compressed_img_3d, k):
    """Calculates Colors, Storage Size (KB), PSNR, and SSIM."""
    h, w, _ = original_img_3d.shape
    
    # 1. Unique Colors (Flatten just to count unique RGB combinations)
    orig_colors = np.unique(original_img_3d.reshape(-1, 3), axis=0).shape[0]
    
    # 2. Storage Math (bits to KB)
    orig_bits = w * h * 24
    orig_kb = orig_bits / (8 * 1024)
    
    b = math.ceil(math.log2(k)) if k > 1 else 1
    comp_bits = (k * 24) + (w * h * b)
    comp_kb = comp_bits / (8 * 1024)
    
    # 3. PSNR (Calculated using float32 to prevent overflow)
    mse = np.mean((original_img_3d.astype(np.float32) - compressed_img_3d.astype(np.float32)) ** 2)
    if mse == 0:
        psnr = 100.0 # Perfect match
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)
        
    # 4. SSIM (Structural Similarity)
    # channel_axis=-1 tells the algorithm that the colors are in the last dimension (R, G, B)
    ssim_value = calculate_ssim(original_img_3d, compressed_img_3d, channel_axis=-1, data_range=255)
        
    return orig_colors, orig_kb, comp_kb, psnr, ssim_value

# --- Visualization Builders (Matplotlib) ---
def plot_color_palette(centroids):
    """Draws a horizontal bar for small K, or a grid for large K (up to 256)."""
    k = len(centroids)
    
    if k <= 50:
        # Standard horizontal strip for small palettes
        palette = np.zeros((1, k, 3), dtype=np.uint8)
        for idx, color in enumerate(centroids):
            palette[0, idx] = color[:3] 
        fig, ax = plt.subplots(figsize=(8, 1))
    else:
        # 2D Grid for large palettes (e.g., 256 becomes a 16x16 grid)
        grid_dim = math.ceil(math.sqrt(k))
        # Create a flat array padded with black to fit the perfect square
        palette = np.zeros((grid_dim * grid_dim, 3), dtype=np.uint8)
        for idx, color in enumerate(centroids):
            palette[idx] = color[:3]
        # Reshape into a 2D image grid
        palette = palette.reshape((grid_dim, grid_dim, 3))
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(palette)
    ax.axis('off')
    return fig

def plot_3d_scatter(pixels, labels, centroids, sample_size=2000):
    # 1. Subsample to prevent freezing
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sampled_pixels = pixels[indices]
        sampled_labels = labels[indices]
    else:
        sampled_pixels = pixels
        sampled_labels = labels

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors_for_plot = centroids[sampled_labels][:, :3] / 255.0 
    
    # Plot the pixel cloud
    ax.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], sampled_pixels[:, 2], 
               c=colors_for_plot, marker='o', s=10, alpha=0.5)

    # --- NEW DYNAMIC CENTROID SIZING ---
    k = len(centroids)
    # If K is small, use massive stars (size 200). If K is large, shrink them to size 40.
    centroid_size = 200 if k <= 50 else 40 

    # Plot the Centroids using the dynamic size
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
               c='black', marker='*', s=centroid_size, label='Centroids')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('RGB Color Space Clustering')
    return fig

# --- Sidebar Configuration ---
st.sidebar.header("Image Selection")

current_file_path = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file_path))
test_image_dir = os.path.join(base_dir, "assets", "test_images")

test_images = ["None"]
if os.path.exists(test_image_dir):
    test_images.extend(os.listdir(test_image_dir))

selected_test_image = st.sidebar.selectbox("Choose a Test Image", test_images)
uploaded_file = st.sidebar.file_uploader("Or Upload Your Own", type=["jpg", "jpeg", "png"])

original_image = None
if uploaded_file is not None:
    original_image = load_and_resize_image_from_upload(uploaded_file)
elif selected_test_image != "None":
    image_path = os.path.join(test_image_dir, selected_test_image)
    original_image = load_and_resize_image_from_path(image_path)

if original_image is not None:
    img_height, img_width, _ = original_image.shape
    
    st.sidebar.header("ML Controls")
    
    mode_display = st.sidebar.selectbox("Feature Mode", ["RGB (Quantization)", "RGB + XY (Segmentation)"])
    actual_mode = "RGB" if "Quantization" in mode_display else "RGB + XY"
    
    operation = st.sidebar.selectbox("Operation", ["Run Clustering", "Find Optimal K (Elbow)"])
    
    if operation == "Run Clustering":
        k_value = st.sidebar.slider("Select K (Colors)", 2, 256, 8)

    run_button = st.sidebar.button("Run ML Engine")

    # --- Main Dashboard Rendering ---
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    if run_button:
        with st.spinner("Crunching the math..."):
            if operation == "Run Clustering":
                
                # Phase 1 & 2: Prep and Process
                pixel_data, img_shape = prep_image_array(original_image, mode=actual_mode)
                labels, centroids = run_kmeans(pixel_data, k=k_value)
                
                # Phase 3: Rebuild Image
                rgb_palette = centroids[:, :3] 
                compressed_pixels = np.uint8(rgb_palette[labels])
                final_image = compressed_pixels.reshape(img_shape) # Rebuild to 3D here!
                
                # Get the original 3D image (strip XY if in 5D mode)
                original_rgb_3d = original_image[:, :, :3]
                
                # Calculate Metrics!
                orig_colors, orig_kb, comp_kb, psnr, ssim_val = calculate_metrics(
                    original_rgb_3d, final_image, k_value
                )
                
                with img_col2:
                    st.subheader(f"Quantized Image (K={k_value})")
                    st.image(final_image, use_container_width=True)
                
                # --- NEW METRICS SCORECARD ---
                st.divider()
                st.header("📈 Compression & Quality Metrics")
                
                # Pass relative widths to give the first columns more horizontal space
                # Ratios: [Colors: 2.5, Storage: 2, Quantized: 1.5, PSNR: 1, SSIM: 1]
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns([2.5, 2, 1.5, 1.5, 1])
                
                with metric_col1:
                    # Back to one line! The expanded column width prevents the "..." cutoff
                    st.metric("Unique Colors", f"{orig_colors:,} ➔ {k_value}")
                with metric_col2:
                    st.metric("Storage Size", f"{orig_kb:,.1f} KB", f"-{orig_kb - comp_kb:,.1f} KB")
                with metric_col3:
                    st.metric("Quantized Size", f"{comp_kb:,.1f} KB")
                with metric_col4:
                    st.metric("PSNR", f"{psnr:.2f} dB", help=">30 dB indicates excellent quality.")
                with metric_col5:
                    st.metric("SSIM", f"{ssim_val:.3f}", help="1.0 is a perfect structural match.")

                # --- VISUALIZATIONS ---
                st.divider()
                st.header("📊 Clustering Visualizations")
                
                vis_col1, vis_col2 = st.columns(2)
                
                with vis_col1:
                    st.subheader("Extracted Color Palette")
                    st.pyplot(plot_color_palette(centroids))
                    
                with vis_col2:
                    st.subheader("3D Pixel Distribution")
                    st.pyplot(plot_3d_scatter(pixel_data, labels, centroids))

            elif operation == "Find Optimal K (Elbow)":
                max_test_k = 32
                pixel_data, img_shape = prep_image_array(original_image, mode=actual_mode)
                wcss = get_wcss_array(pixel_data, max_k=max_test_k)
                
                p1 = np.array([1, wcss[0]])
                p2 = np.array([max_test_k, wcss[-1]])
                distances = []
                
                for i in range(max_test_k):
                    p0 = np.array([i + 1, wcss[i]])
                    dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
                    distances.append(dist)
                
                optimal_k = distances.index(max(distances)) + 1
                optimal_wcss = wcss[optimal_k - 1]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                ax.plot(range(1, max_test_k + 1), wcss, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=4)
                
                ax.plot(optimal_k, optimal_wcss, marker='o', markersize=14, 
                        markeredgewidth=2.5, markeredgecolor='red', markerfacecolor='none')
                
                ax.annotate('Optimal K Threshold\n(Elbow Point)', 
                            xy=(optimal_k, optimal_wcss), 
                            xytext=(optimal_k + 3, optimal_wcss + (wcss[0] * 0.15)),
                            arrowprops=dict(color='red', arrowstyle='->', lw=1.5),
                            fontsize=11, color='red', fontweight='bold')
                
                ax.set_title('Figure 4 - Extended WCSS Variance Curve', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Clusters (K)', fontsize=12)
                ax.set_ylabel('Inertia (WCSS)', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                with img_col2:
                    st.subheader("Elbow Curve Analysis")
                    st.pyplot(fig)
                    st.success(f"**Algorithmic Deduction:** Over an extended {max_test_k}-color test, the maximum curvature point shifts to **K = {optimal_k}**.")