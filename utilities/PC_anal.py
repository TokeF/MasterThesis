from mpl_toolkits.mplot3d import Axes3D # This import registers the 3D projection, but is otherwise unused.
from sklearn.decomposition import PCA
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
sys.path.insert(0, '../utilities')

def PCA_custom(dbdt : np.ndarray, lbl : np.ndarray, name : str):
   #standardize (z-score)
   dbdt_scaler = preprocessing.StandardScaler().fit(dbdt)
   dbdt_std = dbdt_scaler.transform(dbdt)
   X_std = dbdt_std

   #compute principal components
   pca = PCA(n_components=2)
   components = pca.fit_transform(X_std)

   #setup figure
   fig = plt.figure(figsize = (8,8))
   ax = fig.add_subplot(111)
   ax.set_xlabel('Principal Component 1', fontsize = 15)
   ax.set_ylabel('Principal Component 2', fontsize = 15)
   ax.set_title('2D PCA: ' + name, fontsize = 20)

   #Set target ie. inuse or not, color and shape for data
   targets = [0, 1]
   colors = ['r', 'g']
   shapes = ['o', '^']
   #iterate over a tuple (zip), scatter plot 0 and 1
   for target, color, shape in zip(targets,colors, shapes):
      indicesToKeep = lbl == target
      ax.scatter(components[indicesToKeep, 0]
                  , components[indicesToKeep, 1]
                  , c = color
                  , marker = shape
                  , s = 10)
   ax.legend(['coupled', 'non-coupled'])
   ax.grid()
   plt.show()
