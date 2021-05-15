# CIS9650
BRAIN ATTACK (STROKE)

Project description: 
Analyzing the available data set, our group intends to create a model to predict the likelihood of someone having a stroke,
as well as a program for our patients to input their information and get an predicted probability of having a stroke.

Data: 
Kaggle offers an original Stroke Prediction Dataset available for public access at the following web link:
https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/discussion/229886. 
Each observation in the dataset provides the relevant health information of an anonymous individual,
such as gender, age, hypertension, heart disease, marital status, work type, residence type, glucose level,bmi smoking status and occurance of stroke. 
The dataset contains 5,110 observations with 12 attributes. Unknown or N/A imply that the information is either unavailable or not applicable. 
For this analysis, we will use occurrence of stroke as the dependent variable. Occurrence of stroke is implied by 1 for an individual who has a stroke, 0 for an individual who does not have a stroke.

Method: 

Python : numpy,pandas, sklearn,matplotlib,seaborn,os,warnings,dmba;
Statistics: EDA(Exploratory Data Analysis), Oversampling, Naive Bayes

Procedures:
1.Data Pre-Processing: remove null and duplicate values,change data types...
2.Exploratory Data Analysis:
3.Preparation for Model Building: remove outliers,Synthetic Minority Oversampling Technique (SMOTE)...
4.Feature Selection & Comparison of Models: desicion tree, logistic regression, naive bayes, random forest...
5.A Small Program to test people's risk to brain stroke: people input their own data to see their risk to stroke.

(Details for procedure 1-4 is in file 'Stroke_Analysis_and_Model_Building.ipynb', and program for step 5 is in file 'Stroke_Prediction_Program.py '.)

Outcome:
