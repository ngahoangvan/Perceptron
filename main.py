import pandas as pd #loading data in table form
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

#Reading data
data = pd.read_csv("iris.csv")

# Description of data
print("Describing the data: ", data.describe())
# Infomation of data
print("Info of the data:", data.info())

# Visualisation of the dataset
def plotDataTwoFeature(feature_one, feature_two):
    sns.lmplot(feature_one, feature_two,
               data=data,
               fit_reg=False,
               hue="Species",
               scatter_kws={"marker": "D",
                            "s": 50})
    title = '%s vs %s' % (feature_one, feature_two)
    plt.title(title)
    plt.show()

# Parameter is 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
#Example: plot two feature SepalLengthCm and SepalWidthCm
plotDataTwoFeature('PetalLengthCm', 'PetalWidthCm')


"""
# Want to run this code, remove three " in line 34 and 49 and comment code from line 53 to line 67
# This is a different way to define Training Data and Testing Data

# Create our X and y data
# It using for data with spilit 
X = data.drop(labels='Species', axis=1).values
y = data.drop(labels=['SepalLengthCm',
                       'SepalWidthCm',
                      'PetalLengthCm',
                      'PetalWidthCm'],
              axis=1).values

# Split the data into 70% training data and 30% test data. The data will change if we run code again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

"""

# Data X with 4 feature except Species(Class), because I drop it
X_train = pd.read_csv("TrainingDataSet.csv").drop(labels='Species', axis=1).values
X_test = pd.read_csv("TestingDataSet.csv").drop(labels='Species', axis=1).values

"""
# Data X with 2 feature
X_train = pd.read_csv("TrainingDataSet.csv").drop(labels=['Species','SepalLengthCm','SepalWidthCm'], axis=1).values
X_test = pd.read_csv("TestingDataSet.csv").drop(labels=['Species','SepalLengthCm','SepalWidthCm'], axis=1).values
"""

# Data Y with only Species(Class), because I drop(remove) 4 feature 
y_train = pd.read_csv("TrainingDataSet.csv").drop(labels=['SepalLengthCm',
                                                          'SepalWidthCm',
                                                          'PetalLengthCm',
                                                          'PetalWidthCm'],
                                                  axis=1).values
y_test = pd.read_csv("TestingDataSet.csv").drop(labels=['SepalLengthCm',
                                                        'SepalWidthCm',
                                                        'PetalLengthCm',
                                                        'PetalWidthCm'],
                                                axis=1).values

# Train the scaler, which standarizes all the features to have mean=0 and unit variance
sc = StandardScaler()
sc.fit(X_train)

# Apply the scaler to the X training data
X_train_std = sc.transform(X_train)

# Apply the SAME scaler to the X test data
X_test_std = sc.transform(X_test)

# Create a perceptron object with the parameters: 100 iterations (epochs) over the data, and a learning rate of 0.2
ppn = Perceptron(max_iter=100, tol=None, eta0=0.2, random_state=0)

# Train the perceptron
ppn.fit(X_train_std, y_train)

# Apply the trained perceptron on the X data to make predicts for the y test data
y_pred = ppn.predict(X_test_std)

"""
# View the predicted y test data
print(y_pred)

# View the true y test data
print(y_test)
"""

def plotConfusionMatrix():
        cm = confusion_matrix(y_test, y_pred) # The confusion matrix of the model
        ac= accuracy_score(y_test, y_pred)  # The accuracy of the model, 
                                            # which is: 1 - (observations predicted wrong / total observations)
        cm_df = pd.DataFrame(cm,
                     index=['setosa', 'versicolor', 'virginica'],
                     columns=['setosa', 'versicolor', 'virginica'])
        plt.figure(figsize=(5.5, 4))
        sns.heatmap(cm_df, annot=True)
        plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(ac))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

# View the confusion matrix of the model
plotConfusionMatrix()
