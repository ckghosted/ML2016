from util import cluster_data
from s_prob import gen_prob
from l import out_cluster_tags
import pandas as pd

data = pd.read_csv('../data/test.csv')
cluster_data(data, 6)
out_cluster_tags()
gen_prob()
