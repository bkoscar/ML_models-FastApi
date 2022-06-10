# Import the resources for our model
from sklearn.svm import SVC 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
import joblib

"""

Using the dataset previously defined in sklearn called iris, we will applies the machine learning algorithm SVM(Support Vector Machines).

"""



# Load the dataset iris
iris = datasets.load_iris()
# Separate the train values and the test values
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target)

clf = SVC()
# Fit the model
clf.fit(X_train, y_train)
# Save the model with the name iris_model.pkl
joblib.dump(clf,"iris_model.pkl")
