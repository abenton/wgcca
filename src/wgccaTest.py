'''
Test suite for weighted generalized canonical correlation analysis.

Adrian Benton
8/8/2016
'''

import os
import unittest

import wgcca as WGCCA

import numpy as np
import scipy
import scipy.linalg

class TestWeightedGCCA(unittest.TestCase):
  def setUp(self):
    ### Generate sample data with 3 views ###
    self.N  = 1000 # Number of examples
    self.F1 = 50   # Number of features in view 1
    self.F2 = 30
    self.F3 = 40
    self.k  = 5    # Number of latent features
    
    def scale(X):
      X_mean = np.mean(X, axis=0)
      X -= X_mean
      X_std = np.std(X, axis=0)
      X_std[X_std==0.] = 1.0
      
      X /= X_std
      
      return X
    
    def orth(X):
      ''' http://stackoverflow.com/questions/13940056/orthogonalize-matrix-numpy '''
      return X.dot( scipy.linalg.inv(scipy.linalg.sqrtm( X.T.dot(X) )))
    
    # Maps to each view
    W1 = np.random.normal(size=(self.F1, self.k))
    W2 = np.random.normal(size=(self.F2, self.k))
    W3 = np.random.normal(size=(self.F3, self.k))
    W1 = scale(W1)
    W2 = scale(W2)
    W3 = scale(W3)
    
    G  = np.random.normal(size=(self.N, self.k)) # Latent examples
    self.G  = orth(G)
    
    # Observations
    self.V1 = W1.dot(self.G.T).T # N X F1
    self.V2 = W2.dot(self.G.T).T # N X F2
    self.V3 = W3.dot(self.G.T).T # N X F3
    
    ### Write sample data to test file ###
    outFile = open('gcca_test_file.tsv', 'w')
    for i in range(self.N):
      vStrs = [' '.join([str(val) for val in v[i,:]]) for v in [self.V1, self.V2, self.V3]]
      
      # Assume each view is populated from a single document
      outFile.write('%d\t1\t1\t1\t%s\n' % (i, '\t'.join(vStrs)))
    
    outFile.close()
  
  def tearDown(self):
    ''' Remove sample file '''
    
    if os.path.exists('gcca_test_file.tsv'):
      os.remove('gcca_test_file.tsv')
  
  def test_recoverG(self):
    '''
    Test GCCA implementation by seeing if it can recover an orthogonal latent G.
    '''
    
    eps = 1.e-10
    
    Vs = [self.V1, self.V2, self.V3]
    
    wgcca = WGCCA.WeightedGCCA(3, [self.F1, self.F2, self.F3],
                               self.k, [eps, eps, eps], verbose=True)
    wgcca.learn(Vs)
    U1 = wgcca.U[0]
    U2 = wgcca.U[1]
    U3 = wgcca.U[2]
    
    Gprime   = wgcca.apply(Vs, K=None, scaleBySv=False)
    GprimeSv = wgcca.apply(Vs, K=None, scaleBySv=True)
    
    # Rotate G to minimize norm of difference between G and G'
    R, B = scipy.linalg.orthogonal_procrustes(self.G, Gprime)
    normDiff = scipy.linalg.norm(self.G.dot(R) - Gprime)
    
    print 'Recovered G up to rotation; difference in norm:', normDiff
    
    self.assertTrue( normDiff < 1.e-6 )
    self.assertTrue( np.allclose(self.G.dot(R), Gprime) )
  
  def test_ldViews(self):
    ''' Try loading views from file -- ensure they are the same as generated data '''
    
    ids, views = WGCCA.ldViews('gcca_test_file.tsv', [0, 1, 2],
                               replaceEmpty=False, maxRows=-1)
    
    for V, Vprime in zip([self.V1, self.V2, self.V3], views):
      self.assertTrue(np.allclose(V, Vprime))
  
  def test_ldK(self):
    ''' K should be 1 for each view, each example '''
    K = WGCCA.ldK('gcca_test_file.tsv', [0, 1, 2])
    
    self.assertTrue( np.all(K == 1.) )

def main():
  unittest.main()

if __name__ == '__main__':
  main()
