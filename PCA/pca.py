import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components=n_components #no. of principal components to keep
        self.components=None #stores the eigen vectors(new components)
        self.mean=None
    def fit(self,X):
        #compute the mean of the data
        self.mean=np.mean(X,axis=0)
        #subtract mean from each feature
        X=X-self.mean
        #compute covariance matrix of data,function needs features as columns
        cov=np.cov(X.T)
        #calculate eigen values and eigen vectors of covariance matrix
        eigenvalues,eigenvectors=np.linalg.eig(cov)
        #transpose the matrix for easy to sort the eigen vectors
        eigenvectors=eigenvectors.T
        idxs=np.argsort(eigenvalues)[::-1]#sort it in descending order
        eigenvalues=eigenvalues[idxs]
        eigenvectors=eigenvectors[idxs]
        #store the first n eigen vectors(top n)
        self.components=eigenvectors[0:self.n_components]
    def transform(self,X):
        #project data
        X=X-self.mean
        return np.dot(X,self.components.T)
    
if __name__=="__main__":
    from sklearn import datasets
    data=datasets.load_iris()
    X=data.data
    y=data.target
    #fit the data
    pca = PCA(2)
    pca.fit(X)
    x_projected=pca.transform(X)
    print("shape of X :",X.shape)
    print("shape of transformed X:",x_projected.shape)