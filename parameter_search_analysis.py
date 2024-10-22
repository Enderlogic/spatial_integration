import itertools
import pandas
import glob

from tqdm import tqdm

filename_list = glob.glob("results/parameter_search_vae_*")
result = pandas.DataFrame()
for filename in filename_list:
    file = pandas.read_csv(filename)
    result = pandas.concat([result, file])

# dataset_list = ['mouse_spleen_rep1', 'mouse_breast_cancer', 'mouse_spleen_rep2', 'human_lymph_node']
dataset_list = ['human_lymph_node']
learning_rate_list = [1e-3, 1e-4]
zs_dim_list = [16, 32]
zp_dim_list = [16, 32]
hidden_dim1_list = result['hidden_dim1'].unique()
hidden_dim2_list = result['hidden_dim2'].unique()
weight_omics2_list = result['weight_omics2'].unique()
weight_kl_list = [10, 20]
n_neighbors_list = [20, 40]
recon_type_list = ['zinb', 'nb']
heads_list = [3, 1]
epoch_list = result['epoch'].unique()
n_cluster_list = [6, 10]

pss = list(itertools.product(
    *[dataset_list, learning_rate_list, zs_dim_list, zp_dim_list, hidden_dim1_list, hidden_dim2_list,
      weight_omics2_list, weight_kl_list, n_neighbors_list, recon_type_list, heads_list, epoch_list, n_cluster_list]))

summary = pandas.DataFrame(
    columns=['dataset', 'learning_rate', 'zs_dim', 'zp_dim', 'hidden_dim1', 'hidden_dim2', 'weight_omics2', 'weight_kl',
             'n_neighbors', 'recon_type', 'heads', 'epoch', 'n_cluster', 'ari', 'mi', 'nmi', 'ami', 'hom', 'vme',
             'ave_score'])

for i in tqdm(range(len(pss))):
    dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_omics2, weight_kl, n_neighbors, recon_type, heads, epoch, n_cluster = \
        pss[i]
    # single = result[
    #     (result['dataset'] == dataset) & (result['learning_rate'] == learning_rate) & (result['zs_dim'] == zs_dim) & (
    #             result['zp_dim'] == zp_dim) & (result['hidden_dim1'] == hidden_dim1) & (
    #             result['hidden_dim2'] == hidden_dim2) & (result['weight_omics1'] == weight_omics1) & (
    #             result['weight_omics2'] == weight_omics2) & (result['epoch'] == epoch) & (
    #             result['n_cluster'] == n_cluster)][['ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'ave_score']].mean(0)
    single = result[
        (result['dataset'] == dataset) & (result['learning_rate'] == learning_rate) & (result['zs_dim'] == zs_dim) & (
                result['zp_dim'] == zp_dim) & (result['hidden_dim1'] == hidden_dim1) & (
                result['hidden_dim2'] == hidden_dim2) & (result['weight_omics2'] == weight_omics2) & (
                result['weight_kl'] == weight_kl) & (result['n_neighbors'] == n_neighbors) & (
                result['recon_type'] == recon_type) & (result['heads'] == heads) & (result['epoch'] == epoch) & (
                result['n_cluster'] == n_cluster)][['ari', 'mi', 'nmi', 'ami', 'hom', 'vme', 'ave_score']].mean(0)
    summary.loc[len(summary.index)] = [dataset, learning_rate, zs_dim, zp_dim, hidden_dim1, hidden_dim2, weight_omics2,
                                       weight_kl, n_neighbors, recon_type, heads, epoch, n_cluster, single['ari'],
                                       single['mi'], single['nmi'], single['ami'], single['hom'], single['vme'],
                                       single['ave_score']]
summary[(summary['learning_rate'] == 1e-3) & (summary['zs_dim'] == 16) & (summary['zp_dim'] == 16) & (
        summary['hidden_dim1'] == 256) & (summary['hidden_dim2'] == 256)]
a = 1
