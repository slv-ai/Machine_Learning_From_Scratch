import numpy as np

class logisticregression:
    def __init__(self,learning_rate=0.001,n_iters=1000):
        self.lr=learning_rate
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
            #inear model
            linear_model=np.dot(X,self.weights)+self.bias
            #apply sigmoid function to get probabilities
            y_predicted=self._sigmoid(linear_model)
            #compute gradients and update parameters
            dw = (1 / n_samples) * np.dot(X.T,(y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y) 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self,x):
        linear_model=np.dot(x,self.weights)+self.bias
        y_predicted=self._sigmoid(linear_model)
        #convert probabilites to class labels 0 or 1
        y_pred_class=[1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_pred_class)

    def _sigmoid(self,x)                            :
        return 1 / (1 + np.exp(-x))
        
#accuracy function
def accuracy(y_true,y_pred):
        accuracy=np.sum(y_true == y_pred) / len(y_true)
        return accuracy

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    
   #load breast cancer dataset 
    bc_data=datasets.load_breast_cancer()
    X,y=bc_data.data,bc_data.target
    #split the dataset
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=12)
    #train logistic regression model
    lr=logisticregression(learning_rate=0.001,n_iters=1000)
    lr.fit(X_train,y_train)
    #make predictions
    predictions =lr.predict(X_test)
    print("accuracy",accuracy(y_test,predictions))
        