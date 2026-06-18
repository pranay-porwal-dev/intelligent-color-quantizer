# app/ml_engine.py
from sklearn.cluster import MiniBatchKMeans

def run_kmeans(pixel_data, k):
    """
    Runs Mini-Batch K-Means for production-speed clustering.
    """
    # batch_size=2048 means it only looks at 2048 pixels at a time
    # max_iter ensures it doesn't get stuck in an infinite loop
    model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=2048, max_iter=100)
    
    labels = model.fit_predict(pixel_data)
    centroids = model.cluster_centers_
    
    return labels, centroids

def get_wcss_array(pixel_data, max_k=10):
    """
    Calculates the Elbow Method at high speed.
    """
    wcss = []
    for i in range(1, max_k + 1):
        model = MiniBatchKMeans(n_clusters=i, random_state=42, n_init="auto", batch_size=2048, max_iter=100)
        model.fit(pixel_data)
        wcss.append(model.inertia_)
        
    return wcss

# --- Test Block ---
if __name__ == "__main__":
    from image_utils import prep_image
    
    print("Loading image from Phase 1...")
    # Make sure you still have 'sample.jpg' in your main MinorProject folder
    pixels, shape = prep_image("sample.jpg", mode="RGB") 
    
    print("Running ML Engine (K=3)... This might take a few seconds.")
    labels, palette = run_kmeans(pixels, k=3)
    
    print(f"\nSuccess! Found {len(palette)} dominant colors.")
    print("Here is our new color palette (RGB):")
    print(palette)