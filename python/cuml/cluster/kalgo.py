from cuml.cluster.kmeans import KMeans

class KAlgo:
    def __init__(self, **kwargs):
        self.args = kwargs
        
    def algo(self, algo_type):
        
        if algo_type == 'kmeans':
            return KMeans(**self.args)
