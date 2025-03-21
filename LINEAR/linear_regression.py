import numpy as np
def r2_score(y_true,y_pred):
    corr_matrix=np.corrcoef(y_true,y_pred)
    corr=corr_matrix[0,1]
    return corr ** 2

class linearregression:
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
            y_predicted=np.dot(X,self.weights)+self.bias
            #compute gradients & update parameters
            dw=(1 / n_samples) * np.dot(X.T,(y_predicted - y))
            db=(1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self,x):
        y_approxi=np.dot(x,self.weights)+self.bias
        return y_approxi
    
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    def mean_squared_error(y_true,y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
    print(y_test.shape)

    regressor=linearregression(learning_rate=0.001,n_iters=1000)
    regressor.fit(X_train,y_train)
    
    predictions=regressor.predict(X_test)
    print(predictions.shape)

    mse=mean_squared_error(y_test,predictions)
    print("mse",mse)
    
    accuracy=r2_score(y_test,predictions)
    print("accuracy",accuracy)



        

        