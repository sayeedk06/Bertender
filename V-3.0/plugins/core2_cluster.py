#clusters
from sklearn.cluster import KMeans


class Plugin:
    def __init__(self,*args):
        print('\t\n****Cluster plugin activated****\n')

    def initial(self,np_flat_list,label):
        y_pred = KMeans(n_clusters=8, random_state=0).fit_predict(np_flat_list)
        n_clusters_ = len(set(y_pred)) - (1 if -1 in label else 0)
        return y_pred
