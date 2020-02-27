from cuml.cluster.kmeans import KMeans

class KAlgo:
    def __init__(self, handle=None, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=0, random_state=1, init='scalable-k-means++',
                 oversampling_factor=2.0, max_samples_per_batch=1<<15):
        
        self.handle = handle
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.init = init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch

    def algo(self, algo_type):
        
        if algo_type == 'kmeans':
            return KMeans(self.handle, self.n_clusters, self.max_iter, self.tol,
                 self.verbose, self.random_state, self.init,
                 self.oversampling_factor, self.max_samples_per_batch) 
