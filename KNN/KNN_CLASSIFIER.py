import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    distance=np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X):
        y_pred=[self._predict(x)for x in X]
        return np.array(y_pred)
    def _predict(self,x):
        #compute distance b\w x and all examples in training set
        distances=[euclidean_distance(x,x_train)for x_train in self.X_train]
        #sort by distance and return the indices of first k neighbors
        idx=np.argsort(distances)[:self.k]
        #extract the labels
        k_neighbor_labels=[self.y_train[i]for i in idx]
        #return the most common label
        most_common=Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    
if __name__=="__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap=ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true,y_pred):
        accuracy=np.sum(y_true==y_pred) / len(y_true)
        return accuracy
    iris = datasets.load_iris()
    X,y=iris.data,iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)
    k=3

    knnclassifer=KNN(3)
    knnclassifer.fit(X_train,y_train)
    knn_predictions=knnclassifer.predict(X_test)
    print("knn classification accuracy",accuracy(y_test,knn_predictions))

        