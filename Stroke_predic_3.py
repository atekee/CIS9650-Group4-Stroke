import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('healthcare-dataset-stroke-data.csv')
charges = data.drop(["id"],axis=1)
corrDf = charges.apply(lambda x:pd.factorize(x)[0])
corrDf.head()

# find correlation
corr=corrDf.corr()

# heat map
plt.figure(figsize=(10,5),dpi=80)
ax = sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,
               linewidths=0.2,cmap="YlGnBu",annot=True)
plt.title("Correlation between variables")
plt.show()


# plot correlation
factor_data = pd.get_dummies(data.iloc[:,1:])
plt.figure(figsize=(12,5))
factor_data.corr()["stroke"].sort_values(ascending=False).plot(kind="bar")
plt.title("Stroke and variables")
plt.show()
