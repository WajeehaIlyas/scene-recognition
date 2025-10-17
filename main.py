from DataLoader_Resnet import CustomImageDataset
from Resnet_Backbone import Resnet
from utils import get_image_paths, display_results  # your helper function
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import utils
import os
from BOVW import build_vocabulary, get_bags_of_sifts, visualize_sift_keypoints, plot_histogram
from KNN import nearest_neighbor_classify
from SVM import svm_classify

# =========================================================
# Step 0: Setup parameters, paths, and category info
# =========================================================


# 'resnet18'
FEATURE = 'bag of sift'
CLASSIFIER = 'test all'  # options: 'nearest neighbor', 'support vector machine'

data_path = data_path = '/home/wajeeha/Documents/scene-recognition-using-bow-and-resnet-WajeehaIlyas/data/'

categories = np.array([
    'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
    'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
    'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
])

#get image paths is given in utils.py
print('Getting paths and labels for all train and test data\n')
train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path, categories)

#   train_image_paths  1500x1   cell      
#   test_image_paths   705x1    cell           
#   train_labels       1500x1   cell         
#   test_labels        705x1    cell          



# Map string label -> int
categories = sorted(list(set(train_labels)))   # ensure fixed order
class_to_idx = {cat: idx for idx, cat in enumerate(categories)}


# =========================================================
# Step 1: Represent each image with appropriate feature
# =========================================================



print("Using", FEATURE, "representation for images\n")

if FEATURE == 'resnet':

        # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create datasets
    train_dataset = CustomImageDataset(train_image_paths, train_labels, class_to_idx, transform=transform)
    test_dataset  = CustomImageDataset(test_image_paths, test_labels, class_to_idx, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #you can change batch size accordingly
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Num classes:", len(categories))
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    print('getting feature maps from resnet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = Resnet('resnet18').to(device)
    
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = resnet_model(images)  # [batch, 512]
            # store these feature maps and corresponding labels to an array as they will be the input to your classifer
            # these array will also be needed for tsne plot  
    
    #  tsne plot function is given in Resnet_Backbone.py as display_features


elif FEATURE == 'bag of sift':
    VOCAB_SIZE_FINAL = 20
    VOCAB_PATH = f'features/vocab_size_{VOCAB_SIZE_FINAL}.pkl'

    if not os.path.exists('features'):
        os.makedirs('features')

    # Build Vocabulary if it doesn't exist
    if not os.path.exists(VOCAB_PATH):
        print(f'No existing visual word vocabulary found at {VOCAB_PATH}. Computing one from training images\n')
        
        vocab = build_vocabulary(train_image_paths, VOCAB_SIZE_FINAL) 
        
    else:
        print(f'Loading existing visual word vocabulary from {VOCAB_PATH}\n')
     
    train_image_feats = get_bags_of_sifts(train_image_paths, VOCAB_PATH) 
    test_image_feats  = get_bags_of_sifts(test_image_paths, VOCAB_PATH)

    if train_image_feats is None or train_image_feats.size == 0:
        print("CRITICAL ERROR: Feature extraction failed. Cannot proceed to classification/results.")
        exit()

    print("Generating sample visualizations for the report...")
    
    sample_paths = {}
    for path, label in zip(train_image_paths, train_labels):
        if label not in sample_paths:
            sample_paths[label] = path
            
    for label, path in sample_paths.items():
        visualize_sift_keypoints(path)
        
    if train_image_feats is not None:
        plot_histogram(train_image_feats[0], train_labels[0], VOCAB_SIZE_FINAL, 'results/histograms_train_0')
    if test_image_feats is not None:
        plot_histogram(test_image_feats[0], test_labels[0], VOCAB_SIZE_FINAL, 'results/histograms_test_0')
   

elif FEATURE == 'placeholder':
    train_image_feats = []
    test_image_feats = []

else:
    print("Unknown feature type")

# =========================================================
# Step 2: Train Classifier and Predict
# =========================================================

if CLASSIFIER == 'test all':
    
    k_values = [1, 5, 10] 
    
    for k in k_values:
        print(f"\n--- Running k-NN Classifier (k={k}) ---")
        
        predicted_categories = nearest_neighbor_classify(
            train_image_feats, 
            train_labels, 
            test_image_feats, 
            k=k
        )
        
        print(f"\nResults for k-NN (k={k}):")
        display_results(test_labels, categories, predicted_categories)
   
    print("\n--- Running Linear SVM Classifier ---")

    svm_model_path = 'results/linear_svm_bovw.pkl'
    
    predicted_categories = svm_classify(
        train_image_feats, 
        train_labels, 
        test_image_feats,
        model_save_path=svm_model_path
    )
    
    print("\nResults for Linear SVM:")
    display_results(test_labels, categories, predicted_categories)
    
elif CLASSIFIER == 'nearest neighbor':
    predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=5) # Default k=5

elif CLASSIFIER == 'support vector machine':
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

elif CLASSIFIER == 'placeholder':
    # Random guessing for debugging
    predicted_categories = np.random.choice(categories, size=len(test_labels))

else:
    print("Unknown classifier type")


# =========================================================
# Step 3: Display results (The last part of main.py)
# =========================================================

# NOTE: Since the results are now displayed within the 'test all' loop, 
# you should remove or comment out the final standalone display_results() call 
# at the very bottom of the file to prevent errors or double plotting.
# If you leave it, make sure the final `predicted_categories` variable 
# contains the SVM results to be consistent with the assignment structure.
# For simplicity, let's assume you remove the final display call.



# =========================================================
# Step 3: Display results
# =========================================================

## Step 3: Build a confusion matrix and score the recognition system
# You do not need to code anything in this section. 

# If we wanted to evaluate our recognition method properly we would train
# and test on many random splits of the data. You are not required to do so
# for this assignment.

# This function will plot confusion matrix and accuracy of your model
