# **Melanoma Skin Cancer Prediction Using CNN**

## Table of Contents
* [General Info](#general-information)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Project Pipeline](#project-pipeline)
* [Conclusions](#conclusions)
* [Inferences/Observations](#inferences)
* [Acknowledgements](#acknowledgements)


## **General Information**
- This is an assignment in which we use Keras to build a Neural Network.

- In this project, we need to build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

- We use the Melanoma dataset to build the model.

## **Problem Statement**
> In this assignment, you will build a multiclass classification model using a custom convolutional neural network in tensorflow.

**Problem Description**: 
- To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.   
- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant. The data set contains the following diseases:   
1. Actinic keratosis   
2. Basal cell carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented benign keratosis
7. Seborrheic keratosis
8. Squamous cell carcinoma
9. Vascular lesion    
  
NOTE: You don't have to use any pre-trained model using Transfer learning. All the model building process should be based on a custom model.

## Technologies Used
- python - version 3.10
- pandas - version 1.5.2
- notebook - version 6.5.2
- matplotlib - version 3.6.2
- tensorflow - version 2.10.1
- Augmentor - version 0.2.10
- keras -  version 2.11. 0

## Acknowledgements

- This project was based on this tutorial [IIIT-B Upgrad CNN Assignment : DL Spec](https://github.com/ContentUpgrad/Convolutional-Neural-Networks/blob/main/Melanoma%20Detection%20Assignment/Starter_code_Assignment_CNN_Skin_Cancer%20(1).ipynb)

## **Project Pipeline**

1. **Data Reading/Data Understanding** → Defining the path for train and test images.      
2. **Dataset Creation**→ Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.  
3. **Dataset visualisation** → Create a code to visualize one instance of all the nine classes present in the dataset.   
4. **Model Building & training** : Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).   
5. Choose an appropriate optimiser and loss function for model training.  
6. Train the model for ~20 epochs.   
7. Write your findings after the model fit, see if there is evidence of model overfit or underfit.  
8. Choose an appropriate data augmentation strategy to resolve underfitting/overfitting 
9. **Model Building & training on the augmented data:**
 - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
 - Choose an appropriate optimiser and loss function for model training
 - Train the model for ~20 epochs
 - Write your findings after the model fit, see if the earlier issue is resolved or not? **Class distribution: **
 - Examine the current class distribution in the training dataset
 - Which class has the least number of samples?
 - Which classes dominate the data in terms of the proportionate number of samples? Handling class imbalances:
 - Rectify class imbalances present in the training dataset with Augmentor library. Model Building & training on the rectified class imbalance data:
 - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
 - Choose an appropriate optimiser and loss function for model training
 - Train the model for ~30 epochs
 - Write your findings after the model fit, see if the issues are resolved or not?
 The model training may take time to train and hence you can need to use Google colab.

## **Conclusions**

- Those are the summary retrieved from the model, further information can be found in the Notebook.  

- We built the model using CNN with 3 convolutional layers following by a fully connected layer before going to softmax layer.   

1. The first model has the training accuracy is 87% validation accuracy is just around 54%, it seems to overfit
2. The second model has the tensorflow augmenting method applied (`RandomFlip, RandomRotation, RandomZoom`), the overfitting problem is gone but the model is underfit after 20 epochs
3. The final model used the Augmentor library to deal with data imbalancing, the result is better with 94% training accuracy and 83% validation accuracy after 50 epochs 

## **Inferences:**    

* By using **Augmentor library, Data Imbalance Issue is Resolved** & **Overall Accuracy** on training data has **Increased**.

* By adding more CNN layers with **Batch Normalization** & also adding **dropouts**, the **Problem of Overfitting is completely Resolved** now.

* By **tuning the hyperparameter Model**, wrt no of epochs, using appropriate optimizer & loss function, augmentation, class imbalance handling thereby further Drastic Improvement wrt **Increased Performance** was observed.

* **Class Rebalance Really Helped** with **Good Model Performance & Accuracy with No Overfitting** shows evidence issues are resolved.

<!-- This project is Assignment given Under IIT-B & Upgrad In Deep Learning Specialization.-->

### Acknowledgement:
This Project Assignment was part of Curriculum during **EPGP-DataScience AI-ML (Deep Learning Spec) from IIIT-B.**

***@All Rights Reserved*** [**Gurpreet Kaur Jethra**](https://github.com/GURPREETKAURJETHRA)

