import numpy as np

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=5):
    """
    Classifies test images using k-Nearest Neighbors (k-NN) by manually 
    calculating Euclidean distances and performing majority voting.
    
    Args:
        train_image_feats (np.ndarray): Training features (N_train x D).
        train_labels (np.ndarray): Training labels (N_train x 1).
        test_image_feats (np.ndarray): Test features (N_test x D).
        k (int): The number of neighbors to consider for voting.

    Returns:
        np.ndarray: Predicted labels for the test set (N_test x 1).
    """

    print(f"  Implementing custom k-NN with k={k} (Euclidean Distance)...")
    
    num_test = test_image_feats.shape[0]
    predicted_categories = np.empty(num_test, dtype=train_labels.dtype)

    # Prepare for efficient voting by getting integer indices for string labels
    unique_labels, label_indices = np.unique(train_labels, return_inverse=True)

    for i in range(num_test):
        # 1. Calculate squared Euclidean distance to ALL training points
        # ||a - b||^2 = ||a||^2 - 2*(a . b) + ||b||^2. 
        # Numpy handles this efficiently via broadcasting the test vector.
        diff = train_image_feats - test_image_feats[i, :] 
        distances_sq = np.sum(diff**2, axis=1)
        
        # 2. Find the indices of the k-smallest distances
        k_nearest_indices = np.argsort(distances_sq)[:k]
        
        # 3. Get the integer indices of the k nearest labels
        k_nearest_label_indices = label_indices[k_nearest_indices]
        
        # 4. Majority Voting using bincount (fastest way to count occurrences)
        votes = np.bincount(k_nearest_label_indices)
        
        # The predicted label is the one that received the most votes
        predicted_label_index = np.argmax(votes)
        predicted_categories[i] = unique_labels[predicted_label_index]
    
    print("  Prediction complete.")
    return predicted_categories