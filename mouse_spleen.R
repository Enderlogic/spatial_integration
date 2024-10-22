setwd("~/OneDrive - Hong Kong Baptist University/postdoc/spatial_integration")
rm(list = ls())
# load data
library(Seurat)
# replicate 1
rep_1_data <- Read10X_h5('Dataset/mouse_spleen_rep1/GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5')
rep_1_image <- Read10X_Image('Dataset/mouse_spleen_rep1/spatial_rep1')
rep_1 <- CreateSeuratObject(rep_1_data$`Gene Expression`, assay = "RNA", project = "Rep_1")
rep_1_CITE <- CreateSeuratObject(rep_1_data$`Antibody Capture`, assay = "CITE", project = "Rep_1")
rep_1@assays$CITE <- rep_1_CITE@assays$CITE
rep_1$nCount_CITE <- rep_1_CITE$nCount_CITE
rep_1$nFeature_CITE <- rep_1_CITE$nFeature_CITE
rep_1_image@assay <- c("RNA", "CITE")
rep_1_image@key <- "Rep_1"
rep_1@images <- list(Rep_1 = rep_1_image)
# replicate 2
rep_2_data <- Read10X_h5('Dataset/mouse_spleen_rep2/GSE198353_spleen_rep_2_filtered_feature_bc_matrix.h5')
rep_2_image <- Read10X_Image('Dataset/mouse_spleen_rep2/spatial_rep2')
rep_2 <- CreateSeuratObject(rep_2_data$`Gene Expression`, assay = "RNA", project = "Rep_2")
rep_2_CITE <- CreateSeuratObject(rep_2_data$`Antibody Capture`, assay = "CITE", project = "Rep_2")
rep_2@assays$CITE <- rep_2_CITE@assays$CITE
rep_2$nCount_CITE <- rep_2_CITE$nCount_CITE
rep_2$nFeature_CITE <- rep_2_CITE$nFeature_CITE
rep_2_image@assay <- c("RNA", "CITE")
rep_2_image@key <- "Rep_2"
rep_2@images <- list(Rep_2 = rep_2_image)

# MuDataSeurat::WriteH5AD(rep_1, "Dataset/mouse_spleen_rep1/adata_RNA_y.h5ad", assay = "RNA")
# MuDataSeurat::WriteH5AD(rep_1_CITE, "Dataset/mouse_spleen_rep1/adata_ADT_y.h5ad", assay = "CITE")
# MuDataSeurat::WriteH5AD(rep_2, "Dataset/mouse_spleen_rep2/adata_RNA_y.h5ad", assay = "RNA")
# MuDataSeurat::WriteH5AD(rep_2_CITE, "Dataset/mouse_spleen_rep2/adata_ADT_y.h5ad", assay = "CITE")

DefaultAssay(rep_1) = "CITE"
DefaultAssay(rep_2) = "CITE"
rep_1 = NormalizeData(rep_1, assay = "RNA", verbose=FALSE)
rep_1 = NormalizeData(rep_1, normalization.method = "CLR", assay = "CITE", margin = 2, verbose = FALSE)
rep_1 = ScaleData(rep_1, verbose = FALSE)
rep_2 = NormalizeData(rep_2, assay = "RNA", verbose=FALSE)
rep_2 = NormalizeData(rep_2, normalization.method = "CLR", assay = "CITE", margin = 2, verbose = FALSE)
rep_2 = ScaleData(rep_2, verbose = FALSE)

p1 <- SpatialFeaturePlot(rep_1,features = c("CD3","CD4","CD8", "CD19","B220-CD45R", "IgD", "F4-80", "CD163", "CD68"), ncol = 3, images = c("Rep_1"))
p2 <- SpatialFeaturePlot(rep_2,features = c("CD3","CD4","CD8", "CD19","B220-CD45R","IgD", "F4-80", "CD163", "CD68"), ncol = 3, images = c("Rep_2"))

rep_1 = RunPCA(rep_1, features = rownames(rep_1))
rep_1 = FindNeighbors(rep_1, dims = 1:10)
rep_1 = FindClusters(rep_1, resolution = .2, verbose = FALSE)
rep_1 <- RenameIdents(rep_1, '0' = "Macrophage", '1' = "B cell", '2' = "T cell")
plot(SpatialDimPlot(rep_1, images = c("Rep_1")))

rep_2 = RunPCA(rep_2, features = rownames(rep_2))
rep_2 = FindNeighbors(rep_2, dims = 1:10)
rep_2 = FindClusters(rep_2, resolution = .2, verbose = FALSE)
rep_2 <- RenameIdents(rep_2, '0' = "Macrophage", '1' = "B cell", '2' = "T cell")
plot(SpatialDimPlot(rep_2, images = c("Rep_2")))

# output_1 = CreateSeuratObject(rep_1_data$`Gene Expression`, assay = "RNA", project = "Rep_1")
# output_1@assays$CITE <- rep_1_CITE@assays$CITE
# output_1@meta.data$ground_truth = rep_1@active.ident
# MuDataSeurat::WriteH5AD(output_1, "Dataset/mouse_spleen/rep_1_RNA.h5ad", assay = "RNA")
# MuDataSeurat::WriteH5AD(output_1, "Dataset/mouse_spleen/rep_1_ADT.h5ad", assay = "CITE")
# 
# output_2 = CreateSeuratObject(rep_2_data$`Gene Expression`, assay = "RNA", project = "Rep_2")
# output_2@assays$CITE <- rep_2_CITE@assays$CITE
# output_2@meta.data$ground_truth = rep_2@active.ident
# MuDataSeurat::WriteH5AD(output_2, "Dataset/mouse_spleen/rep_2_RNA.h5ad", assay = "RNA")
# MuDataSeurat::WriteH5AD(output_2, "Dataset/mouse_spleen/rep_2_ADT.h5ad", assay = "CITE")

