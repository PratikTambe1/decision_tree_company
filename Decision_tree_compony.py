# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:54:44 2024

@author: Pratik
"""
"""
buisness understanding-
 businesses can gain valuable insights into consumer behavior, market trends, and the effectiveness of their marketing and pricing strategies. 
 This understanding can then inform decision-making processes to optimize sales and maximize profitability.
 
maximize:
    optimizing business strategies for maximizing sales and profitability.

minimize:
    


"""

"""
Data Dictionary:
1.Sales: This column likely represents the sales figures for a particular product or set of products. Understanding the patterns and trends in sales is crucial for assessing the performance of the business.

2.CompPrice: This could refer to the comparative price of the product in comparison to competitors. Analyzing this column helps in understanding how the pricing strategy impacts sales.

3.Income: Income levels of the target demographic can provide insights into the purchasing power of the customers. It helps in segmenting the market and targeting specific income groups.

4.Advertising: This column likely represents the advertising expenditure, which is essential for marketing campaigns. Analyzing the relationship between advertising spending and sales can help evaluate the effectiveness of marketing efforts.

5.Population: The population data could indicate the size of the market or the target audience. Understanding the demographics of the population can assist in market segmentation and targeting.

6.Price: This column could represent the price of the product. Analyzing price elasticity and price sensitivity helps in pricing strategies and understanding consumer behavior.

7.ShelveLoc: This may refer to the shelf location of the product in stores (e.g., premium shelf, regular shelf). Understanding shelf placement is crucial for visibility and sales.

8.Age: Age demographics provide insights into the target market's age distribution. It helps in tailoring products and marketing strategies to different age groups.

9.Education: Education level can influence consumer behavior and preferences. Analyzing this column helps in understanding the educational background of the target market.

10.Urban: This column might indicate whether the store is located in an urban or rural area. Urbanization can impact consumer behavior and preferences.

11.US: This could represent whether the observation is from the US market or not. It helps in analyzing differences in consumer behavior and market trends across regions.

"""

import pandas as pd
df=pd.read_csv('c:/2-dataset/Company_Data.csv')
df.head()
df.columns
#'Sales', 'CompPrice', 'Income', 'Advertising', 'Population', 'Price','ShelveLoc', 'Age', 'Education', 'Urban', 'US'

df.info()
'''
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Sales        400 non-null    float64
 1   CompPrice    400 non-null    int64  
 2   Income       400 non-null    int64  
 3   Advertising  400 non-null    int64  
 4   Population   400 non-null    int64  
 5   Price        400 non-null    int64  
 6   ShelveLoc    400 non-null    object 
 7   Age          400 non-null    int64  
 8   Education    400 non-null    int64  
 9   Urban        400 non-null    object 
 10  US           400 non-null    object 
'''

#there are no null values
df.describe()
"""
            Sales   CompPrice      Income  ...       Price         Age   Education
count  400.000000  400.000000  400.000000  ...  400.000000  400.000000  400.000000
mean     7.496325  124.975000   68.657500  ...  115.795000   53.322500   13.900000
std      2.824115   15.334512   27.986037  ...   23.676664   16.200297    2.620528
min      0.000000   77.000000   21.000000  ...   24.000000   25.000000   10.000000
25%      5.390000  115.000000   42.750000  ...  100.000000   39.750000   12.000000
50%      7.490000  125.000000   69.000000  ...  117.000000   54.500000   14.000000
75%      9.320000  135.000000   91.000000  ...  131.000000   66.000000   16.000000
max     16.270000  175.000000  120.000000  ...  191.000000   80.000000   18.000000
"""
#shelveloc urban and us column have catogerical values
#so we will aplly lable encoding on them
from sklearn.preprocessing import LabelEncoder
le_sh =LabelEncoder()
le_urban=LabelEncoder()
le_us =LabelEncoder()
df['ShelveLoc']=le_sh.fit_transform(df['ShelveLoc'])
df['Urban']=le_urban.fit_transform(df['Urban'])
df['US']=le_us.fit_transform(df['US'])
#df=df.drop(['ShelveLoc','Urban','US'],axis='columns')
df.columns
df.head()
"""
Sales  CompPrice  Income  Advertising  ...  Age  Education  Urban  US
0   9.50        138      73           11  ...   42         17      1   1
1  11.22        111      48           16  ...   65         10      1   1
2  10.06        113      35           10  ...   59         12      1   1
3   7.40        117     100            4  ...   55         14      1   1
4   4.15        141      64            3  ...   38         13      1   0

"""
x=df.drop('Sales',axis=1)
y=df['Sales']


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,  y_test =train_test_split(x,y,test_size=0.2)

#from sklearn.ensemble import RandomForestClassifier
#model=RandomForestClassifier(n_estimators=20)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Example predictions
# Assuming ShelveLoc=2, Urban=1, US=0 for the first example
prediction_1 = model.predict([[115, 95, 5, 110, 117, 26, 20, 0, 1, 1]])

# Assuming ShelveLoc=2, Urban=1, US=1 for the second example
prediction_2 = model.predict([[165, 70, 8, 200, 130, 33, 20, 1, 0, 1]])

print("Prediction 1:", prediction_1)
print("Prediction 2:", prediction_2)















