# GRCGT_LWZ
RGCGT: A high-order feature learning framework for predicting disease-metabolite interaction using residual graph convolution and graph transformer

## 🏠 Overview
![flow chart](https://github.com/LUTGraphGroup/GRCGT_LWZ/assets/109469869/03b68056-4f73-43f7-a063-c5088a279750)


## 🛠️ Dependecies
```
- conda=24.4.0
- Python == 3.12
- pytorch == 2.3.0+cu121
- torch_geometric==2.5.3
- torch_sparse=0.6.18
- numpy == 1.26.4
- pandas == 2.2.2
- scikit-learn==1.5.0
- scipy==1.13.1
- matplotlib==3.9.0
```

## 🗓️ Dataset
```
- disease-metabolite associations: association_DME.xlsx
- disease-microbe associations: association_DMI.xlsx
- microbe-metabolite associations: association_MIME.xlsx
- disease semantic networks based on metapath DMED and DMID: A_DME_D.xlsx and A_DMI_D.xlsx
- metabolite semantic networks based on metapath MEDME and MEMIME: A_DME_ME.xlsx and A_MIME_ME.xlsx 
- disease Gaussian kernel similarity: disease_Gaussian_Simi.xlsx
- disease semantic similarity: disease_Semantic_simi.xlsx
- metabolite functional similarity: metabolite_func_simi.xlsx
- metabolite Gaussian kernel similarity: metabolite_Gaussian_Simi.xlsx
- microbe Gaussian kernel similarities: microbe_Gaussian_Simi_1.xlsx and microbe_Gaussian_Simi_2.xlsx 
```

## 🛠️ Model options
```
--epochs           int     Number of training epochs.                 Default is 1000.
--attn_size        int     Dimension of attention.                    Default is 64.
--attn_heads       int     Number of attention heads.                 Default is 6.
--out_dim          int     Output dimension after feature extraction  Default is 64.
--sampling number  int     enhanced GraphSAGE sampling number         Default is 50.
--dropout          float   Dropout rate                               Default is 0.2.
--slope            float   Slope                                      Default is 0.2.
--lr               float   Learning rate                              Default is 0.001.
--wd               float   weight decay                               Default is 5e-3.

```

## 🎯 How to run?
```
1. The data folder stores various associations and similarities. 
2. The My_code folder for implementing the MAHN model, which specifically includes:
  (1) main.py is used to start the MAHN model and set up parameters.
  (2) train.py is used to implement training and validation, loss function definition, optimizer selection and parameter update.
  (3) model.py is used to build the overall structure of the MAHN model, including different meta-path encoding and bilinear decoder.
  (4) layers.py mainly stores some customized network layers, including multi-head attention layer, semantic level and node level attention layer, etc.
  (5) utils.py mainly realizes data loading, semantic network construction, node sampling and evaluation index calculation, etc.
```
