''' Based on TDHopper's excellent series of notebooks, in particular:
https://github.com/tdhopper/notes-on-dirichlet-processes/blob/master/2015-10-14-collapsed-gibbs-sampling-for-mixture-models.ipynb
'''


import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import choice
import numpy as np
from collections import Counter
from math import sqrt, log, exp

'''plt.scatter(np.array([1,2,3]),np.array([1,2,3]), .8)
plt.show()'''

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
        self.suffstats = {cid: None for cid in self.cluster_ids}
        self.data = data
        self.assignments = [choice(self.cluster_ids) for _ in self.data]
        self.pi = {cid: self.alpha/self.num_clusters for cid in self.cluster_ids}
        self.update_suffstats()
    def update_suffstats(self):
        for cluster_id, N in Counter(self.assignments).items():
            points_in_cluster = [x for x,cid in zip(self.data, self.assignments) if cid==cluster_id]
            mean = float(sum(points_in_cluster))/len(points_in_cluster)
            self.suffstats[cluster_id] = ClusterInfo(mean, N)
    def log_predictive_likelihood(self, data_id, cluster_id):
        cluster_info = ClusterInfo(0,0) if cluster_id=='new' else self.suffstats[cluster_info]
        theta, N = cluster_info.mean, cluster_info.num_pts
        posterior_sigma2 = 1.0/(float(N) / self.cluster_variance + 1.0 / self.hyper_var)
        predictive_mu = posterior_sigma2 * (self.hyper_mean / self.hyper_var + float(N * theta) / self.cluster_variance)
        diff = self.data[data_id] - predictive_mu
        predictive_sigma2 = self.cluster_variance + posterior_sigma2
        return log(exp(-diff*diff/(2*predictive_sigma2)) / sqrt(tau * predictive_sigma2))
    def log_cluster_assign_score(self, cluster_id):
        if cluster_id == 'new': return log(self.alpha)
        current_cluster_size = self.suffstats[cluster_id].num_pts
        return log(current_cluster_size + float(self.alpha)/self.num_clusters)
    def cluster_assignment_distribution(self, data_id):
        scores = {}
        for cid in [self.suffstats.keys()] + ['new']:
            scores[cid] = self.log_predictive_likelihood(data_id, cid)
            scores[cid] += self.log_cluster_assign_score(cid)
        scores = {cid:exp(score) for cid,score in scores.items()}
        normalization = float(sum(scores.values()))
        scores = {cid: score/normalization for cid, score in scores.items()}
        return scores

    def gibbs_step(self):
        pairs = zip(self.data, self.assignments)
        for data_id, (datapoint, cid) in enumerate(pairs): #todo: randomize?
            self.suffstats[cid].remove_datapoint(datapoint)
            scores = self.cluster_assignment_distribution(data_id).items()
            labels, scores = zip(*scores) #insight: zip() is a sort of transpose
            cid = choice(labels, p=scores)
            self.assignments[data_id] = cid
            self.suffstats[cid].add_datapoint(self.data[data_id])

    def plot_clusters(self, title=''):
        gby = pd.DataFrame({'data':self.data, 'assignments':self.assignments}).groupby(by='assignments')['data']
        hist_data = [gby.get_group(cid).tolist() for cid in gby.groups.keys()]
        plt.hist(hist_data, bins=20, histtype='stepfilled',alpha=0.1) #alpha as in opacity
        plt.title(title)
        plt.show()

    def create_cluster(self):
        self.num_clusters += 1
        cluster_id = max(self.suffstats.keys()) + 1
        self.suffstats[cluster_id] = ClusterInfo(0,0)
        self.cluster_ids.append(cluster_id)
        return cluster_id
    def destroy_cluster(self, cid):
        self.num_clusters -= 1
        del self.suffstats[cid]
        self.cluster_ids.remove(cid)
    def prune_clusters(self):
        for cid in self.cluster_ids:
            if self.suffstats[cid].num_pts==0:
                self.destroy_cluster(cid)

data = pd.Series.from_csv('clusters.csv')
S = State(data, 3, 1.0)
S.plot_clusters('0')
S.gibbs_step()
S.plot_clusters('1')
S.gibbs_step()
S.plot_clusters('2')
