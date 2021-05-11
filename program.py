import os
import pandas as pd
import numpy as np
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import percentile
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from dmba import liftChart, gainsChart
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(df.head())

## Missing values
print(df.isnull().sum())
print("\n")

##Duplicates
duplicate = df[df.duplicated()]
print("Duplicate Rows:")

##There are 201 missing values on bmi column but no duplicates in the dataset

##We will handle the missing values in bmi column using using mean() imputation
##because we want to preserve the sample size
df["bmi"].fillna(df["bmi"].mean(),inplace=True)


##print out columns and number of unique values
for col in df.columns:
    print(col, df[col].nunique())
    
print(df["gender"].value_counts())
## we see that there is only 1 row with gender=other, so we drop it out
df.drop(df.index[df["gender"]=="Other"], inplace=True)

##work type children also never worked
df[df["work_type"] == "children"]

#we replace children value with never worked
df["work_type"] = df["work_type"].replace(["children"], "Never_worked")

##next we will rename the residence_type for consistency
##and change age column into integer from float, then define categorical variables
df.rename(columns = {"Residence_type": "residence_type"}, inplace=True)
df["age"] = df["age"].astype("int")

cols = ["stroke", "gender", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type", "smoking_status"]
df[cols] = df[cols].astype("category")

##we also drop id column as it doesn't contribute into our analysis
df.drop(["id"], axis="columns", inplace=True)


##Counting outliers in each numerical column
quant_var = ["bmi", "age", "avg_glucose_level"]
outlier_count = []
print("Number of outliers in each column are as follows: ","\n")

for i in quant_var:
    ##calculate interquartile range
    q25 = np.percentile(df[i], 25)
    q75 = np.percentile(df[i], 75)
    iqr = q75 - q25
    
    ##calculate lower and upper limits
    low_lim = q25 - (1.5 * iqr)
    up_lim = q75 + (1.5 * iqr)
    lst = df[(df[i] < low_lim) | (df[i] > up_lim)].index
    outlier_count.append(len(lst))
    print(i + " ", len(lst))

print("\n","Total outlier count:", sum(outlier_count))


##creating a function to remove outliers
def remove_outliers(data):
    ##calculate interquartile range
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr = q75 - q25
    
    ##calculate lower and upper limits
    low_lim = q25 - (1.5 * iqr)
    up_lim = q75 + (1.5 * iqr)
    
    ##identify and remove outliers
    outliers = []
    for x in data:
        if x < low_lim:
            x = low_lim
            outliers.append(x)
        elif x > up_lim:
            x = up_lim
            outliers.append(x)
        else:
            outliers.append(x)
    return outliers

##removing outliers
df["bmi"] = remove_outliers(df['bmi'])
df["avg_glucose_level"] = remove_outliers(df["avg_glucose_level"])

df.boxplot()
plt.title("After Removing Outliers")
plt.show()


##Encoding
##get dummies and identify predictor and outcome variables
predictors = df.drop(columns = ["stroke"])
outcome = "stroke"

X = pd.get_dummies(predictors, drop_first=True)
y = df[outcome]

##Split validation
train_X, valid_X, train_y, valid_y= train_test_split(X, y, test_size=0.20, random_state=1)


##Oversampling
smote = SMOTE()
train_X, train_y = smote.fit_resample(train_X, train_y)

###distribution of target variable after oversampling
sns.countplot(x = train_y, data = df)
plt.title("Distribution of Stroke after OverSampling")
plt.show()

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(train_X, train_y)
plotDecisionTree(dt, feature_names=train_X.columns, class_names=dt.classes_)


##Get the prediction for both train and test
prediction_train_dt = dt.predict(train_X)
prediction_valid_dt = dt.predict(valid_X)

##Measure the accuracy of the model for both train and test sets
print("Accuracy on training set is:",round(accuracy_score(train_y, prediction_train_dt),2))
print("Accuracy on test set is:",round(accuracy_score(valid_y, prediction_valid_dt ),2))

##Calculating precision, recall and F-measure
p1=round(precision_score(valid_y,prediction_valid_dt),3)
r1=round(recall_score(valid_y,prediction_valid_dt),3)
f1_1=round(f1_score(valid_y,prediction_valid_dt),3)
a1=round(accuracy_score(valid_y, prediction_valid_dt),3)

print("Scores for prediction on validation set")
print("Precision score: ", p1)
print("Recall score: ", r1)
print("f1-score: ", f1_1)
print("Accuracy:",a1)


import statsmodels.api as sm
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features)>0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features+[new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if(min_p_value<significance_level):
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

forward_selection(X,y)

X_for_selected=X[['age',
 'work_type_Never_worked',
 'heart_disease_1',
 'avg_glucose_level',
 'ever_married_Yes',
 'hypertension_1',
 'work_type_Private']]

# splitting the dataset
trainx, validx, trainy, validy= train_test_split(X_for_selected,y,test_size=0.30, random_state=1)

##run naive Bayes
nb = GaussianNB()
nb.fit(trainx,trainy)

##predict class membership
prediction_train_nb=nb.predict(validx)

##predict probabilities
pred_train_prob_nb = nb.predict_proba(validx)

##Prediction accuracy on training and validation set

prediction_train_nb = nb.predict(trainx)
prediction_valid_nb = nb.predict(validx)
print("Accuracy on train is:",accuracy_score(trainy,prediction_train_nb))
print("Accuracy on test is:",accuracy_score(validy,prediction_valid_nb))

##Calculating precision, recall and F-measure on valid
p3=precision_score(validy,prediction_valid_nb)
r3=recall_score(validy,prediction_valid_nb)
f1_3=f1_score(validy,prediction_valid_nb)
a3=accuracy_score(validy, prediction_valid_nb)

print("Precision score: ", p3)
print("Recall score: ", r3)
print("f1-score: ", f1_3)
print("Accuracy:", a3)

print("Enter your own data to test the model: ")
age = int(input("Age:"))
work_type_Never_worked = int(input("Have you never worked?: "))
heart_disease_1 = int(input("Do you have any heart disease?: "))
avg_glucose_level = float(input("What is your avg glucose level? :"))
ever_married_Yes = int(input("Have you every married? :"))
hypertension_1 = int(input("Do you have hypertension? :"))
work_type_Private = int(input("Do you work in private sector? :"))

user_input = [[age, work_type_Never_worked, heart_disease_1, avg_glucose_level,
       ever_married_Yes, hypertension_1, work_type_Private]]

pred_user_output = nb.predict(user_input)
pred_prob_user_output = nb.predict_proba(user_input)
print(pred_user_output)
print(np.round(pred_prob_user_output))




