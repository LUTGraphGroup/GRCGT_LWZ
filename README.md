# GRCGT_LWZ
RGCGT: A high-order feature learning framework for predicting disease-metabolite interaction using residual graph convolution and graph transformer

## 🏠 Overview
![flow chart](https://github.com/LUTGraphGroup/GRCGT_LWZ/assets/109469869/03b68056-4f73-43f7-a063-c5088a279750)


## 🛠️ Dependecies
```
- conda=24.4.0
- Python == 3.12
- pytorch == 2.3.0+cu121
- torch_geometric == 2.5.3
- torch_sparse == 0.6.18
- numpy == 1.26.4
- pandas == 2.2.2
- scikit-learn == 1.5.0
- scipy == 1.13.1
- matplotlib == 3.9.0
- GPU == RTX 2080 Ti(11GB) * 1
- CPU == 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
```

## 🗓️ Dataset
###  Dataset 1: 265 diseases, 2315 metabolites, and 4763 known associations 
```
- disease-metabolite details: disease-metabolite.xlsx
- disease-metabolite associations: association_matrix.csv
- disease similarity matirix: diease_simi_network.csv
- metabolite similarity matirix: metabolite_simi_ntework.csv
- disease initial feature: MeSHHeading2vec.csv
- metabolite initial feature: metabolite_mol2vec.csv
```
###  Dataset 2: 126 diseases, 1405 metabolites, and 2555 known associations 
```
- disease-metabolite details: disease-metabolite.xlsx
- disease-metabolite associations: association_matrix.csv
- disease similarity matirix: diease_simi_network.csv
- metabolite similarity matirix: metabolite_simi_ntework.csv
- disease initial feature: MeSHHeading2vec.csv
- metabolite initial feature: metabolite_mol2vec.csv
```

## 🛠️ Model options
###  training parameters
```
--seed             int     Random seed                                Default is 0.
--epochs           int     Number of training epochs.                 Default is 500.
--weight_decay     float   Weight decay                               Default is 5e-4.
--dropout          float   Dropout rate                               Default is 0.1.
--lr               float   Learning rate                              Default is 0.001.
```

###  training parameters
```
![model_parameter](https://github.com/LUTGraphGroup/GRCGT_LWZ/assets/109469869/90cf3c4d-73f3-415e-9692-85661f2854d5)

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
