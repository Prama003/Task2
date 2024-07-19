
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv('/content/train.csv')

# Check for missing values
missing_values = df.isnull().sum()
print(missing_values)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Visualize distribution of numerical variables
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.histplot(df['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()

# Visualize distribution of categorical variables
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.show()

sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()

# Analyze survival rates by different features
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Step 4: Explore Relationships Between Variables

# Pair plot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare', 'Sex']])
plt.title('Pair Plot')
plt.show()

# Box plot
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Survival by Age')
plt.show()

# Step 5: Identify Patterns and Trends
# Analyzing impact of features on survival
sns.violinplot(x='Survived', y='Age', hue='Pclass', data=df, split=True)
plt.title('Survival by Age and Class')
plt.show()

sns.catplot(x='Sex', y='Survived', hue='Pclass', kind='bar', data=df)
plt.title('Survival by Gender and Class')
plt.show()