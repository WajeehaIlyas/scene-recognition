import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def build_vocabulary(image_paths, vocab_size, s_descriptors=20):
    
    all_descriptors = []
    print("DEBUG: Attempting to create SIFT object...")
    sift = cv2.SIFT_create() # Create SIFT detector/descriptor
    print("DEBUG: SIFT object created successfully.")
    print(f"Extracting up to {s_descriptors} SIFT features per image...")

    print(f"DEBUG: Starting loop over {len(image_paths)} images.")
    
    for path in image_paths:
        print(f"DEBUG: Processing image path: {path}")
        try:
            # Load image in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"DEBUG: Failed to load image. Path: {path}")
                continue

            # Detect keypoints and compute SIFT descriptors
            kp, des = sift.detectAndCompute(img, None)
            
            if des is not None and des.shape[0] > 0:
                # Randomly sample 's_descriptors' features if more are found
                if des.shape[0] > s_descriptors:
                    indices = np.random.choice(des.shape[0], size=s_descriptors, replace=False)
                    des = des[indices, :]
                
                all_descriptors.append(des)
                
        except Exception as e:
            print(f"Error processing image {path}: {e}")
            continue

    # Combine all descriptors into a single large matrix
    if not all_descriptors:
        print("Warning: No descriptors were successfully extracted.")
        return np.empty((vocab_size, 128))
        
    X = np.concatenate(all_descriptors, axis=0).astype(np.float32)
    print(f"Total descriptors collected: {X.shape[0]}")
    
    # Cluster with K-Means
    print(f"Clustering with K-Means (k={vocab_size})...")
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init='auto', verbose=1)
    kmeans.fit(X)

    vocab = kmeans.cluster_centers_  # vocab_size x 128 array
    
    # Save Vocabulary
    filename = f'features/vocab_size_{vocab_size}.pkl'
    try:
        with open(filename, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {filename}")
    except FileNotFoundError:
        print("Warning: Could not save vocabulary. Ensure 'features/' directory exists.")


    return vocab



def get_bags_of_sifts(image_paths, vocab_path):
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}. Run Step 2 first.")
        return None
        
    vocab_size = vocab.shape[0]
    num_images = len(image_paths)
    image_feats = np.zeros((num_images, vocab_size), dtype=np.float32)
    sift = cv2.SIFT_create()
    
    all_sampled_descriptors = [] 
    S_SAMPLE_RATE = 20 
    
    print(f"Processing {num_images} images using vocabulary size: {vocab_size}")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        kp, des = sift.detectAndCompute(img, None)
        
        if des is None or des.shape[0] == 0:
            continue

        if des.shape[0] > S_SAMPLE_RATE:
            indices = np.random.choice(des.shape[0], size=S_SAMPLE_RATE, replace=False)
            des = des[indices, :]
        
        # Save sampled descriptors for the required output
        all_sampled_descriptors.append(des)

        # Assign to Nearest Visual Word (Quantization) ---
        
        distances = np.sum((des[:, np.newaxis, :] - vocab[np.newaxis, :, :])**2, axis=2)
        
        # Find the index (visual word ID) of the minimum distance for each descriptor
        visual_word_indices = np.argmin(distances, axis=1) # M x 1 array
        
        #  Build and Normalize Histogram ---
        histogram, _ = np.histogram(visual_word_indices, bins=range(vocab_size + 1))
        
        # Normalize the histogram (L1 norm is common for BoVW)
        norm_factor = np.sum(histogram)
        if norm_factor > 0:
            image_feats[i, :] = histogram / norm_factor
        else:
            image_feats[i, :] = histogram

    try:
        # Save a combined list of sampled descriptors from all images
        combined_descriptors = np.concatenate(all_sampled_descriptors, axis=0)
        output_filename = 'features/all_sampled_descriptors_for_report.pkl'
        with open(output_filename, 'wb') as f:
            pickle.dump(combined_descriptors, f)
        print(f"Successfully saved combined sampled descriptors to {output_filename}")
    except Exception as e:
        print(f"Error saving sampled descriptors: {e}")
        
    return image_feats

def visualize_sift_keypoints(image_path, output_dir='results/sample_sift_keypoints_per_class'):
    """Draws detected SIFT keypoints on an image and saves the result."""
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image at {image_path}")
        return
        
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    
    # Draw keypoints (default color is red)
    img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Determine save path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    class_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f'sift_keypoints_{class_name}_{image_name}')
    
    # Save the image
    cv2.imwrite(save_path, img_kp)
    print(f"Saved SIFT visualization to {save_path}")

    return save_path

def plot_histogram(histogram_vector, category_label, vocab_size, output_dir='results/histograms'):
    """Visualizes the Bag of Words histogram for a single image feature vector."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(12, 5))
    
    # Histogram vector should be a 1D array
    if histogram_vector.ndim > 1:
        histogram_vector = histogram_vector.flatten()
        
    # Plot the histogram as a bar chart
    plt.bar(range(vocab_size), histogram_vector, width=1.0)
    
    plt.title(f'BoVW Histogram for Category: {category_label} (Vocab Size: {vocab_size})')
    plt.xlabel('Visual Word Index (Cluster ID)')
    plt.ylabel('Normalized Frequency')
    plt.xticks(np.arange(0, vocab_size, step=vocab_size // 10 if vocab_size >= 10 else 1)) # show some ticks
    plt.grid(axis='y', alpha=0.75)
    
    # Save the plot
    save_path = os.path.join(output_dir, f'histogram_{category_label}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram visualization to {save_path}")
    
    return save_path