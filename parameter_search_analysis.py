import itertools
import pandas
import glob

from tqdm import tqdm

filename_list = glob.glob("results/parameter_search_mmvaeplus_*")
result = pandas.DataFrame()
for filename in filename_list:
    file = pandas.read_csv(filename)
    result = pandas.concat([result, file])

# dataset_list = ['mouse_spleen_rep1', 'mouse_breast_cancer', 'mouse_spleen_rep2', 'human_lymph_node_rep1']
dataset_list = result['dataset'].unique()
learning_rate_list = [1e-3, 1e-4]
zs_dim_list = [16, 32]
zp_dim_list = [16, 32]
hidden_dim1_list = [256, 512]
hidden_dim2_list = [256, 512]
weight_kl_list = [10, 20]
heads_list = [3, 1]
epoch_list = result['epoch'].unique()

pss = list(itertools.product(
    *[dataset_list, learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list, weight_kl_list,
      heads_list, epoch_list]))

summary = pandas.DataFrame(
    columns=['dataset', 'learning_rate', 'zs_dim', 'zp_dim', 'hidden_dim1', 'hidden_dim2', 'weight_kl',
             'heads', 'epoch', 'n_cluster', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'ave_score', 'moran', 'jaccard1',
             'jaccard2', 'jaccard'])

for i in tqdm(range(len(pss))):
    dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_kl, heads, epoch = pss[i]
    if 'spleen' in dataset:
        n_cluster_list = [5, 3]
    elif 'lymph_node' in dataset:
        n_cluster_list = [6, 10]
    elif 'breast_cancer' in dataset:
        n_cluster_list = [5]
    elif 'thymus' in dataset:
        n_cluster_list = [8]
    elif 'brain' in dataset:
        n_cluster_list = [18]
    else:
        raise Exception('Data not recognized')
    for n_cluster in n_cluster_list:
        single = result[(result['dataset'] == dataset) & (result['learning_rate'] == learning_rate) & (
                result['zs_dim'] == zs_dim) & (result['zp_dim'] == zp_dim) & (result['hidden_dim1'] == hidden_dim1) & (
                                result['hidden_dim2'] == hidden_dim2) & (result['weight_kl'] == weight_kl) & (
                                result['heads'] == heads) & (result['epoch'] == epoch) & (
                                result['n_cluster'] == n_cluster)][
            ['ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'ave_score', 'moran', 'jaccard1', 'jaccard2', 'jaccard']].mean(0)
        summary.loc[len(summary.index)] = [dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_kl,
                                           heads, epoch, n_cluster, single['ari'], single['mi'], single['nmi'],
                                           single['ami'], single['hom'], single['vme'], single['ave_score'],
                                           single['moran'], single['jaccard1'], single['jaccard2'], single['jaccard']]
summary[(summary['learning_rate'] == 1e-3) & (summary['zs_dim'] == 16) & (summary['zp_dim'] == 16) & (
        summary['hidden_dim1'] == 256) & (summary['hidden_dim2'] == 256)]
a = 1
