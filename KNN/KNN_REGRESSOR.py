import numpy as np

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KNNREGRESSOR:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X):
        y_pred=[self._predict(x)for x in X]
        return np.array(y_pred)
    def _predict(self,x):
        #compute distances between x and all in training set
        distances=[euclidean_distance(x,x_train)for x_train in self.X_train]
        #sort by distance and return indices of first k neighbors
        idx=np.argsort(distances)[:self.k]
        #extract the target labels of neighbor training samples
        k_neighbor_values=[self.y_train[i]for i in idx]
        #return the mean of target values\
        return np.mean(k_neighbor_values)
    
if __name__=="__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    #load dataset
    diabetes=datasets.load_diabetes()
    X,y=diabetes.data,diabetes.target
    #split the dataset
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    k=3
    knnregressor=KNNREGRESSOR(k=k)
    #train the model
    knnregressor.fit(X_train,y_train)
    #make predictions on dataset
    predictions=knnregressor.predict(X_test)
    #evaluate the model using MSE
    mse=mean_squared_error(y_test,predictions)
    print(f"knn_regressor mean squared error:{mse}")