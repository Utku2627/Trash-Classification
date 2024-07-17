**# Trash-Classification**

This is a Trash Classification project with CNN model in Python.


## **Introduction**
The main goal of this project is to classify 6 types of waste using deep learning techniques, specifically Convolutional Neural Networks (**CNN**).

To achieve this classification, I used a previously utilized dataset and trained my own model with this dataset after creating it. I experimented with different parameters to obtain various results and searched for the optimum solution.

This project is designed to meet specific requirements using a conda environment. You can download the **.yaml** file containing all the necessary dependencies from the project files.


#**Model Deatils**


##** Model Architecture**

The Convolutional Neural Network (CNN) model used in this project consists of:
- 7 Convolutional Layers
- 3 MaxPooling Layers
  

##**Importing the Dataset**

The dataset is imported from GaryThung's repository. You can download the dataset from there or you can download it by downloading .data1.rar and .data2.rar files and merging them in a file that named as ".data" 


##**Installation**

To implement the project you need to create a conda environment. 

  conda create --name <my-env>  //Prompt to create an environment. (  <my-env> is your environment name  )

  You can simply download the "Anaconda Navigator" and import the "Environment.yaml" file that is given in the files.

  After downlading Navigator you can import the entire environment to your computer.

  After succesfully imported the environment all your depencies should be satisfied.


##**CudaTest**

This class is not necessary to run the project. It checks the torch if it using the GPU or not.


##**Utils**
This class contains some functions that are necessary to execute the train.py
  

##** Data Preprocessing**

The photos in the dataset have dimensions of 512x284. Therefore, when setting hyperparameters, the input should be adjusted according to the dataset.

Before executing the train.py file you should execute the ImagePreprocessing.py file because this file splits the dataset and make it usable for train.py.

To successfully execute the ImagePreprocessing.py file you may need to change the file paths, or you can simply set the file path of the dataset as "C:\TrashClassification\.data".

After executing the ImagePreprocessing.py you are ready to execute train.py and getting results.


# **Some Results**







![Accuracy over Epochs](https://github.com/user-attachments/assets/14b4f47b-0553-4b90-adbe-6ff3108d48ed)

This is the Accuracy over Epochs graphic, as it shows while epochs are reaching higher numbers our accuracy is getting higher but after some epochs this difference can be ignored so it will be unnecessary to keep the model running. 





![Screenshot 2024-07-17 134638](https://github.com/user-attachments/assets/9fe3e887-e012-404d-8f5e-2c544e76fac8)

This is the Loss over Epoch graphic, as it shows while epoch are reaching higher numbers our Validation Loss and Train Loss getting lower for a  while, but after some epochs our Validation Loss getting higher again. It means that our optimum epoch number is between 7 and 10.





![Screenshot 2024-07-17 134717](https://github.com/user-attachments/assets/516a4a61-7a0f-41e3-8bf5-91a1c160a400)

This is the confusion matrix of our model. It shows us the accuracy of our model for each label. For example our model has 93% accuracy for plastic and 18% accuracy for trash. 

Our overall accuracy is 0.7344 for this results it can be change with respect to hyperparameters and model.

We can use a bigger dataset to train the model better. It will increase the accuracy.

We can use other models to get higher accuracy.

We can find the optimum hyperparameter values to get higher accuracy.



