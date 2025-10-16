import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def build_vocabulary(image_paths, vocab_size, s_descriptors=400):
    
    all_descriptors = []
    sift = cv2.SIFT_create() # Create SIFT detector/descriptor
    print(f"Extracting up to {s_descriptors} SIFT features per image...")
    
    for path in image_paths:
        try:
            # Load image in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
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
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init='auto', verbose=0)
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
    return image_feats
