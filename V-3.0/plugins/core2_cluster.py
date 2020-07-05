#clusters
from sklearn.cluster import KMeans


class Plugin:
    def __init__(self,*args):
        print('\t\n****Cluster plugin activated****\n')

    def initial(self,np_flat_list,label):
        y_pred = KMeans(n_clusters=8, random_state=0).fit(np_flat_list)
        centers = y_pred.cluster_centers_
        labels = y_pred.labels_
        # n_clusters_ = len(set(y_pred)) - (1 if -1 in label else 0)
        return labels, centers
