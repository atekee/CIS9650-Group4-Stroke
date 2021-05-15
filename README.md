# CIS9650
BRAIN ATTACK (STROKE)
Group 4: Xin Bian, Bingxuan Hu, Atabay Kadiroglu, Qiaoyi Tan, Wenyi Wang, Zixin Ye

Project description: 
According to WHO, stroke is the second leading cause of dealth and major cause of disability worldwide. We intend to create a progarm that can help people monitor their risks of getting a stroke.

Our solution is to:
Step 1) create a classification model to predict whether an individual will have a stroke or not based on available features from our dataset; and then 
Step 2) design a program where users can input their information and get a predicted probability of having a stroke.

Instructions:
1. Check the source code for Solution Step 1: Stroke_Analysis_and_Model_Building.ipynb
2. Run our interface and program for Step 2: Stroke_Prediction_Program.py 

Data: 
Kaggle offers an original Stroke Prediction Dataset available for public access at the following web link:
https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/discussion/229886. 
Each observation in the dataset provides the relevant health information of an anonymous individual,
such as gender, age, hypertension, heart disease, marital status, work type, residence type, glucose level,bmi smoking status and occurance of stroke. 
The dataset contains 5,110 observations with 12 attributes. Unknown or N/A imply that the information is either unavailable or not applicable. 
For this analysis, we will use occurrence of stroke as the dependent variable. Occurrence of stroke is implied by 1 for an individual who has a stroke, 0 for an individual who does not have a stroke.

Programming Languages: 
Jupyter Notebook
Python: numpy, pandas, sklearn, imblearn

Statistics Preparation: 
1.Data Pre-Processing: remove null and duplicate values,change data types
3.Exploratory Data Analysis
4.Preparation for Model Building: remove outliers,Synthetic Minority Oversampling Technique (SMOTE)

Model Selection
1.Feature Selection
2.Comparison of Models: desicion tree, logistic regression, naive bayes, random forest
3.Naive Bayes return the best restuls

Program for Users：
1. The user interface allows user to input their personal data and the program would analyze the inputs to determine how likely for that person to get a stroke. 
2. It helps people to be alerted, and take preemptive measures to lower the risk of having a stroke
