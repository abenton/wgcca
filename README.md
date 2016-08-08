# wgcca
Python Implementation of Weighted Generalized Canonical Correlation Analysis as described in 
"Learning Multiview Embeddings of Twitter Users". Benton A, Arora R, and Dredze M. ACL 2016.

Tested with

+ Python 2.7
+ scipy 0.17.0
+ numpy 1.10.4

Test suite:

    python src/wgccaTest.py

Sample call to learn 5-dimensional WGCCA model (first two views weighted twice as much as second two):

    python src/wgcca.py --input resources/sample_wgcca_input.tsv.gz --output wgcca_embeddings.npz --model wgcca_model.pickle --k 5 --kept_views 0 1 2 3 --weights 1.0 1.0 0.5 0.5 --reg 1.e-8 1.e-8 1.e-8 1.e-8    python src/wgcca.py

* Input format can be grokked from: `resources/sample_wgcca_input.tsv`
* WGCCA model saved to: `wgcca_model.pickle`
* WGCCA embeddings saved to: `wgcca_embeddings.npz`

WeightedGCCA methods
----
* `_compute`: look at this if you want to know how embeddings are computed
* `learn`: entrypoint for learning WeightedGCCA model from training set
* `apply`: entrypoint for extracting embeddings from new data

Please contact *adrian dot author1_surname at gmail dot com* if you have any
questions/suggestions/concerns/comments.
