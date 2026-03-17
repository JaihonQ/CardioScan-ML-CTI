from sklearn.neural_network import MLPClassifier

class NeuralNetworkModel(MLPClassifier):
    def __init__(self):
        super().__init__(
            hidden_layer_sizes = 350 ,
            max_iter = 1000 , 
            activation = 'relu'
        ) 
    def fit_nn_model(self,x,y):
        super().fit(x,y)
        print('Model Trained.')
    def predict_nn_model(self,x):
        return super().predict(x)