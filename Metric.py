import scanpy as sc
import numpy as np
import squidpy as sq
from sklearn.preprocessing import OneHotEncoder

def Moran(adata):

    # 计算空间邻居信息
    sq.gr.spatial_neighbors(adata)

    # 计算 Moran's I 分数
    # 可以指定基因的名称，或计算所有基因的 Moran's I
    sq.gr.spatial_autocorr(adata, mode='moran')
    return adata

# 计算Jaccard相似度
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_jaccard_similarity(adata_mod1, adata_mod2, k=50):
    """
    计算两个嵌入空间的邻居集合之间的Jaccard相似度。

    Parameters:
    - embedding1: numpy.ndarray，形状为 (n_spots, n_genes)，表示第一个模态的矩阵
    - embedding2: numpy.ndarray，形状为 (n_spots, n_features)，表示第二个embedding的矩阵
    - k: int，表示最近邻的数量，默认值为50

    Returns:
    - jaccard_similarities: numpy.ndarray，形状为 (n_samples,)，每个spot的Jaccard相似度
    """

    # 将这些数据封装为 AnnData 对象
    # adata_mod1 = sc.AnnData(modality)
    # adata_mod2 = sc.AnnData(embedding)


    # 在模态和embedding中分别计算最近邻
    sc.pp.neighbors(adata_mod1, use_rep= 'X', n_neighbors=k)
    sc.pp.neighbors(adata_mod2, use_rep= 'X', n_neighbors=k)

    # 初始化用于存储所有spots Jaccard相似度的列表
    jaccard_similarities = []

    # 对于所有spots，逐一提取邻居集合并计算Jaccard相似度
    num_spots = adata_mod1.shape[0]

    for i in range(num_spots):
        # 提取模态1和模态2中第i个spot的邻居集合
        neighbors_mod1 = np.nonzero(adata_mod1.obsp['distances'][i].toarray())[1]
        neighbors_mod2 = np.nonzero(adata_mod2.obsp['distances'][i].toarray())[1]

        # 提取邻居集合并转换为set，以便计算Jaccard相似度
        N_im = set(neighbors_mod1)
        N_ie = set(neighbors_mod2)
        J_i = jaccard_similarity(N_im, N_ie)
        jaccard_similarities.append(J_i)

    # 将结果转换为 numpy 数组
    jaccard_similarities = np.array(jaccard_similarities)

    # 打印统计信息
    print(f"Average Jaccard similarity: {jaccard_similarities.mean():.4f}")
    print(f"Standard deviation: {jaccard_similarities.std():.4f}")

    return jaccard_similarities.mean()