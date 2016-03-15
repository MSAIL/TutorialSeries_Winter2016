import pandas as pd
from numpy.random import choice
from collections import Counter
from math import sqrt, log, exp

tau = 6.283185307179586476925286766559

class ClusterInfo:
    def __init__(self, mean, num_pts):
        self.mean, self.num_pts = mean, num_pts
    def add_datapoint(self, x):
        self.mean = (self.mean*self.num_pts + x) / (self.num_pts + 1)
        self.num_pts += 1
        #return ClusterInfo((self.mean*self.num_pts+x)/(self.num_pts+1), self.num_pts+1)
    def remove_datapoint(self, x):
        self.mean = (self.mean*self.num_pts - x) / (self.num_pts - 1)
        self.num_pts -= 1
        #return ClusterInfo((self.mean - self.num_pts+x)/(self.num_pts - 1), self.num_pts - 1)

class State:
    def __init__(self, data, num_clusters, alpha):
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.cluster_ids = range(self.num_clusters)
        self.cluster_variance = 0.01
        self.hyper_mean = 0.0; self.hyper_var = 1.0
        self.suffstats = {cid: None for ci in self.cluster_ids}
        self.data = data
        self.assignments = [choice(self.cluster_ids) for _ in self.data]
        self.pi = {cid: self.alpha/self.num_clusters for cid in self.cluster_ids}
    def update_suffstats(self):
        for cluster_id, N in Counter(self.assignments).iteritems():
            points_in_cluster = [x for x,cid in zip(self.data, self.assigments) if cid==cluster_id]
            mean = float(sum(points_in_cluster))/len(points_in_cluster)
            state.suffstats[cluster_id] = ClusterInfo(mean, N)
    def log_predictive_likelihood(self, data_id, cluster_id):
        theta, N = self.suffstats[cluster_id].mean, self.suffstats[cluster_id].num_pts
        posterior_sigma2 = 1.0/(float(N) / self.cluster_variance + 1.0 / self.hyper_var)
        predictive_mu = posterior_sigma2 * (self.hyper_mean / self.hyper_var + float(N * theta) / self.cluster_variance)
        diff = self.data[data_id] - mu
        predictive_sigma2 = self.cluster_variance + posterior_sigma2
        return log(exp(-diff*diff/(2*predictive_sigma2)) / sqrt(tau * predictive_sigma2))
    def log_cluster_assign_score(self, cluster_id):
        current_cluster_size = self.suffstats[cluster_id].N
        return log(current_cluster_size + float(self.alpha)/self.num_clusters)
    def cluster_assignment_distribution(self, data_id):
        scores = {}
        for cid in self.suffstats.keys():
            scores[cid] = self.log_predictive_likelihood(data_id, cid)
            scores[cid] += self.log_cluster_assign_score(cid)
        normalization = float(sum(scores.values())
        scores = {cid: score/normalization for cid, score in scores.iteritems()}
        return scores

    def gibbs_step(self):
        pairs = zip(self.data, self.assignments)
        for data_id, (datapoint, cid) in enumerate(pairs): #todo: randomize?
            self.suffstats[cid].remove_datapoint(datapoint)
            scores = self.cluster_assignment_distribution(data_id)).items()
            labels, scores = zip(*scores) #insight: zip() is a sort of transpose
            cid = choice(labels, p=scores)
            self.assignments[data_id] = cid
            self.suffstats[cid].add_datapoint(self.data[data_id])

    def plot_clusters(self):
        gby = pd.DataFrame({'data':self.data, 'assignments':self.assignments}).groupby(by='assignments')['data']
        hist_data = [gby.get_group(cid).tolist() for cid in gby.groups.keys()]
        plt.hist(hist_data, bins=20, histtype='stepfilled',alpha=0.1) #alpha as in opacity

S = State()
