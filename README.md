### Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
### EXPLANATION

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

### ALGORITHM
STEP 1

Read the given Data.

STEP 2

Get the information about the data.

STEP 3

Detect the Outliers using IQR method and Z score.

STEP 4

Remove the outliers.

STEP 5

Plot the datas using Box Plot.

### PROGRAM

Developed by : NITHYAA SRI S S

Registration Number : 21222230100


```

import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)

```

# OUTPUT

DATASET FOR BHP_CSV



DATASET HEAD(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989063-14bdde0a-5135-4aa7-b9b8-93a4f616045b.png)

DATASET DESCRIBE(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989266-64ac4d6a-f569-4275-a1b8-ef4e65c01dc4.png)

DATASET INFO(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989461-4f5c96bb-7b40-46ad-8d1e-8435a2de7112.png)

DATASET NULL VALUES(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989690-7839803a-776f-4b3f-9a55-071529591f93.png)

DATASET SHAPE WITH OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989802-5cbdd8a5-0677-4ea3-baeb-a1cc491f4145.png)

DATASET BOXPLOT WITH OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/119122478/226989950-e0a6220e-f291-40f6-bbb2-49e69d89ed96.png)

DATASET WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/119122478/226990067-031960e2-0653-4972-aa8d-e6d2f5f48815.png)

![image](https://user-images.githubusercontent.com/119122478/226990187-d0b3c45a-7c65-472b-a376-8e2da79d98b3.png)

DATASET SHAPE WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/119122478/226990398-72ce7796-0182-4f15-b3e0-07585f4514f6.png)


DATASET BOXPLOT WITHOUT OUTLIERS(BHP)

![image](https://user-images.githubusercontent.com/119122478/226990513-6ad72c55-5dc3-45f5-924c-70d60eb9617b.png)


DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![image](https://user-images.githubusercontent.com/119122478/226990568-381d8e66-282c-443f-8e79-ffa06aaffb30.png)


DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
 
![image](https://user-images.githubusercontent.com/119122478/226990681-c08837e9-af17-42b6-a748-1836fe1a5698.png)


DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)

![image](https://user-images.githubusercontent.com/119122478/226990778-6ad63f6c-cacb-45d0-ba4a-f76ec2bb11b6.png)


DATASET FOR WEIGHT_HEIGHT_CSV

![image](https://user-images.githubusercontent.com/119122478/226990885-b4465adb-0319-435d-85d2-4e06b48aa65b.png)


DATASET HEAD(WEIGHT_HEIGHT)

![image](https://user-images.githubusercontent.com/119122478/226991016-bf777891-16b8-468a-96bb-86223333b9b1.png)


DATASET INFO(WEIGHT_HEIGHT)



DATASET DESCRIBE(WEIGHT_HEIGHT)

![OUTPUT](dataset18.png)

DATASET NULL VALUES(WEIGHT_HEIGHT)

![OUTPUT](dataset19.png)

DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)

![OUTPUT](dataset%2020.png)

DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![OUTPUT](dataset%2021.png)

![OUTPUT](dataset%2022.png)

DATASET SHAPE(WEIGHT_HEIGHT)

![OUTPUT](dataset%2023.png)

DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![OUTPUT](dataset24.png)


### RESULT

The given datasets are read and outliers are detected and are removed using IQR and z-score methods.











