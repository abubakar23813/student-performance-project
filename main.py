import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Load Data
df=pd.read_csv('Students.csv')
print(df.head())


#checking Missing values
print(df.isnull().sum())


#drop missing values
df = df.dropna()


# create   total and average column
df['total'] = df['math score'] + df['reading score'] + df['writing score']
df['average'] = df['total'] / 3

print(df.head())

# find top students
df = df.sort_values(by=['average'], ascending=False)
print(df.head())

#Histogram
df['math score'].hist()
plt.title('Math score Distribution')
plt.xlabel('Marks')
plt.ylabel('number of students')
plt.show()

# Scatter plot
plt.scatter(df['reading score'], df['math score'])
plt.title('reading score vs Math score ')
plt.xlabel('reading score')
plt.ylabel('math score')
plt.show()


# finding top 5 students using Bar Chart
top5 = df.sort_values(by=['average'], ascending=False).head()
plt.bar(top5.index.astype(str), top5['average'])
plt.title('Top 5 Students')
plt.xlabel('Students Index')
plt.ylabel('Average Marks')
plt.show()


#Machine Learning Model
X = df[[ 'reading score', 'writing score']]
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print('Model Accuracy:', score)

# predicting the score
new_data = pd.DataFrame([[80,85]], columns=['reading score', 'writing score'])
prediction = model.predict(new_data)
print('predicted math score:',prediction)