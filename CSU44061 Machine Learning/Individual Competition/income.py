import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction import DictVectorizer
import pickle
import sys

print(sys.version)

def correlate(dataset):
    corrmat = dataset.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)

    k = 12 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'income')['income'].index
    cm = np.corrcoef(dataset[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

def scale(dataset):
    names = dataset.columns
    scaler = MinMaxScaler()
    scaled_dataset = scaler.fit_transform(dataset)
    scaled_dataset = pd.DataFrame(scaled_dataset, columns=names)
    return scaled_dataset

def encodeOneHot(dataset):
    dataset_genders = pd.get_dummies(dataset['Gender'], prefix = 'gender', drop_first=True)
    dataset = pd.concat([dataset, dataset_genders], axis=1)
    dataset = dataset.drop('Gender', axis=1)

    dataset_colors = pd.get_dummies(dataset['Hair Color'], prefix = 'color', drop_first=True)
    dataset = pd.concat([dataset, dataset_colors], axis=1)
    dataset = dataset.drop('Hair Color', axis=1)

    dataset_degree = pd.get_dummies(dataset['University Degree'], prefix = 'degree', drop_first=True)
    dataset = pd.concat([dataset, dataset_degree], axis=1)
    dataset = dataset.drop('University Degree', axis=1)

    dataset_jobs = pd.get_dummies(dataset['Profession'], prefix = 'job', drop_first=True)
    dataset = pd.concat([dataset, dataset_jobs], axis=1)
    dataset = dataset.drop('Profession', axis=1)

    dataset_country = pd.get_dummies(dataset['Country'], prefix = 'country', drop_first=True)
    dataset = pd.concat([dataset, dataset_country], axis=1)
    dataset = dataset.drop('Country', axis=1)

    return dataset


def encodeLabel(dataset):
    dataset.Gender = LabelEncoder().fit_transform(dataset.Gender)
    dataset['Hair Color'] = LabelEncoder().fit_transform(dataset['Hair Color'])
    dataset['University Degree'] = LabelEncoder().fit_transform(dataset['University Degree'])
    dataset.Profession = LabelEncoder().fit_transform(dataset.Profession)
    dataset.Country = LabelEncoder().fit_transform(dataset.Country)

    return dataset

def predictSVR(X, Y):

    # Spliting into 80% for training set and 20% for testing set so we can see our accuracy
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    model = SVR()

    model.fit(X_train,Y_train)
    y_pred = model.predict(x_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    df1 = df.head(25)
    print(df1.head(25))

    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def predictLR(X, Y):

    # Spliting into 80% for training set and 20% for testing set so we can see our accuracy
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    regressor = LinearRegression()  
    regressor.fit(X_train, Y_train)

    y_pred = regressor.predict(x_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    df.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def preprocess(dataset):
    #Get rid of vague data
    dataset = dataset[dataset['University Degree'] != '0']
    dataset = dataset[dataset['Gender'] != '0']
    dataset = dataset[dataset['income'] > 0]
    dataset = dataset[dataset['Gender'] != 'unknown']
    dataset = dataset.dropna()
    dataset['income'] = dataset['income'].astype('int')

    # dataset = dataset.groupby('Profession').filter(lambda x : len(x)>160)
    # dataset = dataset.groupby('Age').filter(lambda x : len(x)>100)
    # dataset = dataset.groupby('Country').filter(lambda x : len(x)>5)

    # Int 
    dataset['Year of Record'] = dataset['Year of Record'].astype('int')
    dataset['Age'] = dataset['Age'].astype('int')

    Y = dataset['income']
    dataset = dataset.drop('income', 1)

    print(dataset.head())
    dataset = encodeOneHot(dataset)
    dataset = scale(dataset)


    # correlate(dataset)
    print(dataset.head())

    # # dataset.groupby('University Degree').income.mean().plot(kind='bar')
    # # dataset.groupby('Profession').income.mean().plot(kind='bar')
    # # sns.relplot(x="University Degree", y="income", hue='Age', data=dataset)
    # # plt.show()
    # # ENCODE
    # # dataset = encodeLabel(dataset)

    # Int 
    dataset['Year of Record'] = dataset['Year of Record'].astype('int')
    dataset['Age'] = dataset['Age'].astype('int')


    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Instance')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Size of City')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Wears Glasses')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Body')))]
    # dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='job')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='country')))]

    print(dataset.shape)
    print(dataset.describe)

    X = dataset

    return X, Y

def preprocessPredict(dataset):
    print(dataset.head())
    dataset = dataset.drop('income', 1)
    #Get rid of vague data
    dataset = dataset[dataset['University Degree'] != '0']
    dataset = dataset[dataset['Gender'] != '0']
    dataset = dataset[dataset['Gender'] != 'unknown']
    dataset = dataset.dropna()

    # dataset = dataset.groupby('Profession').filter(lambda x : len(x)>10)
    # dataset = dataset.groupby('Age').filter(lambda x : len(x)>20)
    # dataset = dataset.groupby('Country').filter(lambda x : len(x)>5)
    # Int 
    dataset['Year of Record'] = dataset['Year of Record'].astype('int')
    dataset['Age'] = dataset['Age'].astype('int')

    dataset = encodeOneHot(dataset)
    dataset = scale(dataset)


    # correlate(dataset)
    print(dataset.head())

    # # dataset.groupby('University Degree').income.mean().plot(kind='bar')
    # # dataset.groupby('Profession').income.mean().plot(kind='bar')
    # # sns.relplot(x="University Degree", y="income", hue='Age', data=dataset)
    # # plt.show()
    # # ENCODE
    # # dataset = encodeLabel(dataset)

    # Int 
    dataset['Year of Record'] = dataset['Year of Record'].astype('int')
    dataset['Age'] = dataset['Age'].astype('int')


    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Instance')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Size of City')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Wears Glasses')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='Body')))]
    # dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='job')))]
    dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='country')))]

    print(dataset.shape)
    print(dataset.describe)

    X = dataset

    return X



def main():
    dataset = pd.read_csv('/Users/mario/Documents/CSU44061 Machine Learning/Individual Competition/data/tcd ml 2019-20 income prediction training (with labels).csv')
    X, Y = preprocess(dataset)
    predictSVR(X, Y)
    dataset = pd.read_csv('/Users/mario/Documents/CSU44061 Machine Learning/Individual Competition/data/tcd ml 2019-20 income prediction test (without labels).csv')

    X = preprocessPredict(dataset)

    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    result = loaded_model.predict(X)
    result.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')

if __name__ == "__main__":
    main()






