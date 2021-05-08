
import pandas as pd
import numpy as np
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# drop duplicates
data = data.drop_duplicates()

#data clean
data = data.drop(data[data["gender"]=="Other"].index)
data["smoking_status"].replace("Unknown",np.nan,inplace=True)
data["smoking_status"].fillna(data["smoking_status"].mode()[0],inplace=True)
data["bmi"].fillna(data["bmi"].mean(),inplace=True)

'''**********************data analysis*************************'''
# group by stroke = 0 and stroke = 1
dt0 = data[data['stroke']==0]
print('stroke=0：')
print(dt0.describe())

dt1 = data[data['stroke']==1]
print('stroke=1：')
print(dt1.describe())

#percentage of gender in stroke=0、1
dt = data.groupby(['stroke','gender'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['gender']=='Male')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,gender=Male ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['gender']=='Female')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,gender=Female ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['gender']=='Male')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,gender=Male ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['gender']=='Female')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,gender=Female ratio:%.2f%%' % (t * 100))

#percentage of hypertension stroke=0、1
dt = data.groupby(['stroke','hypertension'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['hypertension']==0)]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,hypertension=0 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['hypertension']==1)]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,hypertension=1 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['hypertension']==0)]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,hypertension=0 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['hypertension']==1)]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,hypertension=1 ratio:%.2f%%' % (t * 100))

#percentage of hear_disease in stroke=0、1
dt = data.groupby(['stroke','heart_disease'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['heart_disease']==0)]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,heart_disease=0 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['heart_disease']==1)]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,heart_disease=1 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['heart_disease']==0)]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,heart_disease=0 ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['heart_disease']==1)]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,heart_disease=1 ratio:%.2f%%' % (t * 100))

#percentage of ever_married in stroke=0、1
dt = data.groupby(['stroke','ever_married'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['ever_married']=='Yes')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,ever_married=Yes ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['ever_married']=='No')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,ever_married=No ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['ever_married']=='Yes')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,ever_married=Yes ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['ever_married']=='No')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,ever_married=No ratio:%.2f%%' % (t * 100))

#工作类型分别在stroke=0、1的占比
dt = data.groupby(['stroke','work_type'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['work_type']=='Private')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,work_type=Private ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['work_type']=='Self-employed')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,work_type=Self-employed ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['work_type']=='Govt_job')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,work_type=Govt_job ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['work_type']=='children')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,work_type=children ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['work_type']=='Never_worked')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,work_type=Never_worked ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['work_type']=='Private')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,work_type=Private ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['work_type']=='Self-employed')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,work_type=Self-employed ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['work_type']=='Govt_job')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,work_type=Govt_job ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['work_type']=='children')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,work_type=children ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['work_type']=='Never_worked')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,work_type=Never_worked ratio:%.2f%%' % (t * 100))

#户口类型分别在stroke=0、1的占比
dt = data.groupby(['stroke','Residence_type'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['Residence_type']=='Urban')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,Residence_type=Urban ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['Residence_type']=='Rural')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,Residence_type=Rural ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['Residence_type']=='Urban')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,Residence_type=Urban ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['Residence_type']=='Rural')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,Residence_type=Rural ratio:%.2f%%' % (t * 100))

#percentage of smoke_status in stroke=0、1
dt = data.groupby(['stroke','smoking_status'])['id'].count().reset_index()
t = dt[(dt['stroke']==0) & (dt['smoking_status']=='formerly smoked')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,smoking_status=formerly smoked ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['smoking_status']=='never smoked')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,smoking_status=never smoked ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==0) & (dt['smoking_status']=='smokes')]['id'].sum()/dt[dt['stroke']==0]['id'].sum()
print('stroke=0,smoking_status=smokes ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['smoking_status']=='formerly smoked')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,smoking_status=formerly smoked ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['smoking_status']=='never smoked')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,smoking_status=never smoked ratio:%.2f%%' % (t * 100))
t = dt[(dt['stroke']==1) & (dt['smoking_status']=='smokes')]['id'].sum()/dt[dt['stroke']==1]['id'].sum()
print('stroke=1,smoking_status=smokes ratio:%.2f%%' % (t * 100))


