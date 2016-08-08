'''
Implementation of weighted generalized canonical correlation analysis as
described in:

@inproceedings{benton2016learning,
  title={Learning multiview embeddings of twitter users},
  author={Benton, Adrian and Arora, Raman and Dredze, Mark},
  year={2016},
  organization={ACL}
}

Adrian Benton
8/6/2016
'''

#from theano import Tensor as T

import cPickle, gzip, os, sys, time

import numpy as np
import scipy
import scipy.linalg

import argparse

class WeightedGCCA:
  '''
  Weighted generalized canonical correlation analysis (WGCCA).
  Implemented with batch SVD.
  '''
  
  def __init__(self, V, F, k, eps, viewWts=None, verbose=True):
    self.V = V # Number of views
    self.F = F # Number of features per view
    self.k   = k # Dimensionality of embedding we want to learn
    self.eps = eps # Regularization for each view
    self.W  = viewWts if viewWts else [1. for v in range(V)] # How much we should weight each view -- defaults to equal weighting
    
    self.U = None # Projection from each view to shared space
    self.G = None # Embeddings for training examples
    self.G_scaled = None # Scaled by SVs of covariance matrix sum
    
    self.verbose = verbose
  
  def _compute(self, views, K=None):
    '''
    Compute G by first taking low-rank decompositions of each view, stacking
    them, then computing the SVD of this matrix.
    '''
    
    # K ignores those views we have no data for.  If it is not provided,
    # then we use all views for all examples.  All we need to know is
    # K^{-1/2}, which just weights each example based on number of non-zero
    # views.  Will fail if there are any empty examples.
    if K is None:
      K = np.ones((views[0].shape[0], len(views)))
    
    # We do not want to count missing views we are downweighting heavily/zeroing out, so scale K by W
    K = K.dot(np.diag(self.W))
    Ksum = np.sum(K, axis=1)
    
    # If we have some missing rows after weighting, then make these small & positive.
    Ksum[Ksum==0.] = 1.e-8
    
    K_invSqrt = scipy.sparse.dia_matrix( ( 1. / np.sqrt( Ksum ), np.asarray([0]) ), shape=(K.shape[0], K.shape[0]) )
    
    # Left singular vectors for each view along with scaling matrices
    As = []
    Ts = []
    Ts_unnorm = []
    
    N = views[0].shape[0]
    
    # Take SVD of each view, to calculate A_i and T_i
    for i, (eps, view) in enumerate(zip(self.eps, views)):
      A, S, B = scipy.linalg.svd(view, full_matrices=False, check_finite=False)
      
      # Find T by just manipulating singular values.  Matrices are all diagonal,
      # so this should be fine.
      
      S_thin = S[:self.k]
      
      S2_inv = 1. / (np.multiply( S_thin, S_thin ) + N * eps)
      
      T = np.diag(
            np.sqrt(
              np.multiply( np.multiply( S_thin, S2_inv ), S_thin )
            )
          )
      
      # Keep singular values
      T_unnorm = np.diag( S_thin + eps )
      
      # Keep the left singular vectors of view j
      As.append(A[:,:self.k])
      Ts.append(T)
      Ts_unnorm.append(T_unnorm)
      
      if self.verbose:
        print 'Decomposed data matrix for view %d' % (i)
    
    # In practice M_tilde may be really big, so we would
    # like to perform this SVD incrementally, over examples.
    M_tilde = K_invSqrt.dot( np.bmat( [ np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts) ] ) )
    Q, R = scipy.linalg.qr( M_tilde, mode='economic')
    
    # Ignore right singular vectors
    U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)
    
    self.G = Q.dot(U[:,:self.k])
    
    # Unnormalized version of G -> captures covariance between views
    M_tilde = K_invSqrt.dot( np.bmat( [ np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts_unnorm) ] ) )
    Q, R = scipy.linalg.qr( M_tilde, mode='economic')
    
    # Ignore right singular vectors
    U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)
    
    self.lbda = lbda
    self.G_scaled = self.G.dot(np.diag(self.lbda[:self.k]))
    
    if self.verbose:
      print 'Decomposed M_tilde / solved for G'
    
    self.U = [] # Mapping from views to latent space
    self.U_unnorm = [] # Mapping, without normalizing variance
    
    # Now compute canonical weights
    for idx, (eps, f, view) in enumerate(zip(self.eps, self.F, views)):
      R = scipy.linalg.qr(view, mode='r')[0]
      Cjj_inv = np.linalg.inv( (R.transpose().dot(R) + N * eps * np.eye( f )) )
      pinv = Cjj_inv.dot( view.transpose() )
      
      self.U.append(pinv.dot( self.G ))
      self.U_unnorm.append(pinv.dot( self.G_scaled ))
      
      if self.verbose:
        print 'Solved for U in view %d' % (idx)
    
    #import pdb; pdb.set_trace()
  
  def learn(self, views, K=None):
    '''
    Learn WGCCA embeddings on training set of views.
    '''
    
    self._compute(views, K)
    return self
  
  def apply(self, views, K=None, scaleBySv=False):
    '''
    Extracts WGCCA embedding to new set of examples.  Maps each view with $U_i$ and 
    
    If scaleBySv is true, then does not normalize variance of each canonical
    direction.  This corresponds to GCCA-sv in "Learning multiview embeddings of twitter users."
    Applying WGCCA to a single view with this set to true is equivalent to PCA.
    '''
    
    Us        = self.U_unnorm if scaleBySv else self.U
    projViews = []
    
    N = views[0].shape[0]
    
    if not K:
      K = np.ones((N, self.V)) # Assume we have data for all views/examples
    
    for U, v in zip(Us, views):
      projViews.append( v.dot(U) )
    
    # Get mean embedding from all views, weighting each view appropriately
    
    weighting = np.multiply(K, self.W) # How much to weight each example/view
    
    # If we don't have weight on any view, embedding for that example will be NaN
    denom = weighting.sum(axis=1).reshape((N, 1))
    
    # Weight each view
    weightedViews = []
    for i, v in enumerate(projViews):
      weightedViews.append( np.multiply(weighting[:,i].reshape((N, 1)), v) )
    Gsum = sum(weightedViews)
    
    # Normalize by sum of view weights to get mean
    Gprime = np.multiply(Gsum, 1./denom)
    
    return Gprime

def fopen(p, flag='r'):
  ''' Opens as gzipped if appropriate, else as ascii. '''
  if p.endswith('.gz'):
    if 'w' in flag:
      return gzip.open(p, 'wb')
    else:
      return gzip.open(p, 'rb')
  else:
    return open(p, flag)

def ldViews(inPath, viewsToKeep, replaceEmpty=True, maxRows=-1):
  '''
  Loads dataset for each view.
  Input file is tab-separated: first column ID, next k columns are number of documents that
  go into each view, other columns are the views themselves.  Features in each view are
  space-separated floats -- dense.
  
  replaceEmpty: If true, then replace empty rows with the mean embedding for each view
  
  Returns list of IDs, and list with an (N X F) matrix for each view.
  '''
  
  N = 0 # Number of examples
  
  V = 0 # Number of views
  FperV = [] # Number of features per view
  
  f = fopen(inPath) 
  flds = f.next().split('\t')
  
  V = (len(flds) - 1) / 2
  
  # Use all views
  if not viewsToKeep:
    viewsToKeep = [i for i in range(numViews)]
  
  for fld in flds[1+V:]:
    FperV.append(len(fld.split()))
  f.close()
  
  f  = fopen(inPath)
  
  flds = f.next().strip().split('\t')
  F  = [len(fld.split()) for fld in flds[(1+V):]]
  #V  = len(F)
  N += 1
  
  for ln in f:
    N += 1
  f.close()
  
  data = [np.zeros((N, numFs)) for numFs in F]
  ids  = []
  
  f = fopen(inPath)
  for lnidx, ln in enumerate(f):
    if (maxRows > 0) and (lnidx >= maxRows):
      break
    
    if not lnidx % 10000:
      print 'Reading line: %dK' % (lnidx/1000)
    
    #if (lnidx == 5000):
    #  break
    
    flds = ln.split('\t')
    ids.append(int(flds[0]))
    
    viewStrs = flds[(1+V):]
    
    for idx, viewStr in enumerate(viewStrs):
      for v in viewStr.split():
        data[idx:]
      vals = [float(v) for v in viewStr.split()]
      for fidx, v in enumerate(viewStr.split()):
        try:
          data[idx][lnidx,fidx] = float(v)
        except Exception, e:
          print e
          import pdb; pdb.set_trace()
    
  f.close()
  
  # Replace empty rows with the mean for each view
  if replaceEmpty:
    means = [np.sum(d, axis=0)/np.sum(1. * (np.abs(d).sum(axis=1) > 0.0)) for d in data]
    for i in range(N):
      for vIdx in viewsToKeep:
        if sum([np.sum(data[vIdx][i,:]) for vIdx in viewsToKeep]) == 0.:
          data[vIdx][i,:] = means[vIdx]
  
  return ids, data

def ldK(p, viewsToKeep):
  '''
  Returns matrix K, indicating which (example, view) pair is missing.
  
  p: Path to data file
  viewsToKeep: Indices of views we want to keep.
  '''
  
  numLns   = 0
  
  f = fopen(p)
  for ln in f:
    numViews = ln.count('\t')/2
    numLns += 1
  f.close()
  
  # Use all views
  if not viewsToKeep:
    viewsToKeep = [i for i in range(numViews)]
  
  K = np.ones((numLns, len(viewsToKeep)))
  
  # We keep count of number of tweets and infos collected in each view, and first field is the user ID.
  # Just zero out those views where we did not collect any data for that view
  f = fopen(p)
  for lnIdx, ln in enumerate(f):
    flds = ln.split('\t')
    for idx, vidx in enumerate(viewsToKeep):
      count = int(flds[vidx+1])
      if count < 1:
        K[lnIdx,idx] = 0.0
  f.close()
  
  return K

def main(inPath, outPath, modelPath, k, keptViews=None, weights=None, regs=None, scaleBySv=False, saveGWithModel=False):
  '''
  Read in views, learn GCCA mapping to latent space
  
  inPath:    Tab-separated file containing views.
  outPath:   Where to write out WGCCA embeddings as compressed numpy file.
  modelPath: Where to write out pickled WGCCA model.
  k:         Dimensionality of WGCCA embeddings.
  keptViews: Which views to learn model on.  Passing None defaults to learning on all views.
  weights:   Weighting for each view.  Passing None defaults to equal weighting.
  scaleBySv: Scale training embeddings by singular values of sum of covariance matrices.
  saveGWithModel: Whether training set embeddings are pickled with the model.
  '''
  
  ids, views = ldViews(inPath, keptViews, replaceEmpty=False, maxRows=-1)
  K          = ldK(inPath, keptViews)
  
  # Default to equal weighting
  if not weights:
    weights = [1.0 for v in views]
  if not regs:
    regs = [1.e-8 for v in views]
  
  wgcca = WeightedGCCA(len(views), [v.shape[1] for v in views],
                       k, regs, weights, verbose=True)
  wgcca = wgcca.learn(views, K)
  
  # Save model
  if modelPath:
    if not saveGWithModel:
      wgcca.G = None
      wgcca.G_scaled = None
    
    modelFile = fopen(modelPath, 'wb')
    cPickle.dump(wgcca, modelFile)
    modelFile.close()
  
  # Save training set embeddings
  if outPath:
    G = wgcca.apply(views, K, scaleBySv)
    np.savez_compressed(outPath, ids=ids, G=G)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', default=None, help='tab-separated view data')
  parser.add_argument('--output', default=None,
                      help='path where to save embeddings for each example')
  parser.add_argument('--model', default=None,
                      help='path where to save pickled WGCCA model')
  parser.add_argument('--k', default=None, help='dimensionality of embeddings')
  parser.add_argument('--kept_views', default=None,
                      type=int, nargs='+',
                      help='indices of views to learn model over.  Defaults to using all views.')
  parser.add_argument('--weights', default=None,
                      type=float, nargs='+',
                      help='how much to weight each view in the WGCCA objective -- defaults to equal weighting')
  parser.add_argument('--reg', default=None,
                      type=float, nargs='+',
                      help='how much regularization to add to each view\'s covariance matrix.  Defaults to 1.e-8 for each view.')
  parser.add_argument('--scale_by_sv', default=False, action='store_true',
                      help='scale columns of G by singular values of sum of covariance matrices; corresponds to GCCA-sv in "Learning Multiview Embeddings of Twitter Users"')
  parser.add_argument('--save_g_with_model', default=True, action='store_false',
                      help='save training set embeddings with WGCCA model (consumes space)')
  args = parser.parse_args()
  
  if not (args.model or args.output):
    print 'Either --model or --output must be set...'
  elif not args.k:
    print 'Need to set k, width of learned embeddings'
  else:
    main(args.input, args.output, args.model,
         args.k, args.kept_views, args.weights,
         args.scale_by_sv, args.save_g_with_model)
