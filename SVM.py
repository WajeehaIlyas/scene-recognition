import numpy as np
from sklearn.svm import LinearSVC
import pickle 
import os

def svm_classify(train_image_feats, train_labels, test_image_feats, model_save_path='results/linear_svm_bovw.pkl'):
    
    print("  Training Linear SVM...")
    
    # Initialize the Linear SVM classifier
    classifier = LinearSVC(C=1.0, dual='auto', random_state=42, max_iter=2000)

    # Train the classifier
    classifier.fit(train_image_feats, train_labels)

    # Save the trained model to a file using pickle
    os.makedirs(os.path.dirname(model_save_path) or '.', exist_ok=True)
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(classifier, f)
        
    print(f"  SVM Model saved to: {model_save_path}")

    # Predict the labels for the test features
    print("  Predicting test labels...")
    predicted_categories = classifier.predict(test_image_feats)
    
    return predicted_categories