import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
import os
import glob

def get_image_paths(data_path, categories):
    train_image_paths, train_labels = [], []
    train_base_path = data_path + 'train/train/' 

    for cat in categories:
        search_path = train_base_path + cat + '/*'
        
        if os.path.isdir(train_base_path + cat):
            pass
        
        imgs = glob.glob(search_path)
            
        train_image_paths = train_image_paths + imgs
        train_labels = train_labels + [cat]*len(imgs)

    test_image_paths, test_labels = [], []
    test_base_path = data_path + 'test/test/'
    test_categories = os.listdir(test_base_path)
    
    for cat in test_categories: 
        if cat not in categories: 
            continue 
            
        search_path = test_base_path + cat + '/*' 
        
        imgs = glob.glob(search_path)
            
        test_image_paths = test_image_paths + imgs
        test_labels = test_labels + [cat]*len(imgs)

    return np.array(train_image_paths), np.array(test_image_paths), np.array(train_labels), np.array(test_labels)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def perf_measure(y_actual, y_hat):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        elif y_actual[i]==y_hat[i]==0:
            TN += 1
        elif y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return [TP, FP, TN, FN]


def display_results(test_labels, categories, predicted_categories):

    df = pd.DataFrame(columns= ['Category']+list(categories))

    cols = ['Category']+['TP', 'FP', 'TN', 'FN']
    df = pd.DataFrame(columns= cols)
    
    # 1. (No change needed here) Calculate metrics per category
    for el in categories:
        temp_y_test = (test_labels == el).astype(int)
        temp_preds = (predicted_categories == el).astype(int)
        row = [el]+ perf_measure(temp_y_test, temp_preds)
        df = df._append(pd.Series(row, index=cols), ignore_index=True)
    print(df, '\n\n')
    
    # Identify the unique categories ACTUALLY present in the test set (y_true)
    present_categories = np.unique(test_labels)
    
    # Create the mapping for ONLY the categories that need to be plotted
    class_to_int = {cat: i for i, cat in enumerate(present_categories)}

    # Convert the label arrays using ONLY the present categories
    temp_test_labels = np.array([class_to_int[lab] for lab in test_labels])
    temp_predicted_categories = np.array([class_to_int.get(lab, -1) for lab in predicted_categories])
    
    # If a predicted category is not in the test set, it's irrelevant and gets a placeholder (-1)
    # Filter out these placeholder predictions to ensure array sizes match
    mask = temp_predicted_categories != -1
    temp_predicted_categories = temp_predicted_categories[mask]
    temp_test_labels = temp_test_labels[mask]
    
    # Reset categories to be the PRESENT_CATEGORIES for plotting
    class_names = present_categories # <--- Use only the categories present in y_true
    
    class_names = present_categories # Use only present categories
    
    
    # 3. Define class_names as the PRESENT categories (y_true) for the plotter.
    present_categories = np.unique(test_labels) # Categories present in y_true
    class_names = present_categories 
    
    # 4. Call the plotter.
    plot_confusion_matrix(test_labels, predicted_categories, classes=class_names)
    fig = plt.gcf()
    fig.show()
    
    # 5. Fix f1_score call to use string labels and specify all 15 categories.
    all_categories = np.array(categories)
    f1 = f1_score(y_pred=predicted_categories, y_true=test_labels, labels=all_categories, average='macro') # <--- FIX f1_score HERE
    print('f1 score: ', f1)
    
    return