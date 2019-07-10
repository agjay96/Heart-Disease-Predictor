from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        #raise NotImplementedError
        self.features=features
        self.labels=labels

    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        #raise NotImplementedError
        pred=[]
        lab=[]
        #print("length of features",len(features))
        #if(f is not None):
        for f in features:
            lab=self.get_k_neighbors(f)
            zero=lab.count(0)
            one=lab.count(1)

            if(zero>one):
                pred.append(0)
            else:
                pred.append(1)

        return pred 
       

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        #raise NotImplementedError
        lab=[]
        distances=[]
        
        for train in self.features:
            distances.append(self.distance_function(train, point))

        dict2=[]
        ind = []
        dict2=np.argsort(distances)

        z=int(self.k)
        ind=dict2[:z]
        for m in ind:
            lab.append(self.labels[m])

        return lab


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
