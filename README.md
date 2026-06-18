# Intelligent Color Quantization & Semantic Segmentation Platform 🎨

An enterprise-grade, modular computer vision application that bridges Digital Image Processing (DIP) and Unsupervised Machine Learning. The platform implements optimized Mini-Batch K-Means clustering to transition high-resolution images from a 24-bit Direct Color model to an Indexed Color model (Look-Up Tables), dropping storage footprints by up to 83% while preserving textural visual integrity.

##  System Architecture
The codebase strictly enforces the **Separation of Concerns (SoC)** principle across three distinct modular layers:
- `app/app.py`: Reactive GUI frontend powered by Streamlit, featuring smart `@st.cache_data` state management.
- `app/ml_engine.py`: Mathematical black-box executing Scikit-Learn's `MiniBatchKMeans`.
- `app/image_utils.py`: Rapid data prep pipeline handling BGR-to-RGB conversion, matrix flattening, and spatial normalization.

##  Mathematical Foundations

### 1. Objective Function (Within-Cluster Sum of Squares)
The core engine optimizes color buckets by minimizing global variance via Euclidean distance geometry:
$$WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2$$

### 2. Information Theory Storage Allocation
True hardware storage depth reduction is computed using the ceiling function of a base-2 logarithm:
$$b = \lceil \log_2(K) \rceil \text{ bits per pixel}$$

## 🚀 Key Features
- **3D Quantization Mode (RGB):** Standard color depth minimization calculated natively inside a Cartesian 3D space.
- **5D Segmentation Mode (RGB + XY):** Feature augmentation that appends normalized spatial coordinates, forcing regional object isolation.
- **The Elbow Method:** Automates heuristic loop tracking from $K=1$ to $10$ to systematically find the optimal palette boundary.
- **transparent Analytics:** Displays real-time calculations tracking Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).

## 🛠️ Setup Instructions
1. Clone the repository:
   ```bash
   git clone [https://github.com/pranay-porwal-dev/intelligent-color-quantizer.git](https://github.com/pranay-porwal-dev/intelligent-color-quantizer.git)
   cd intelligent-color-quantizer