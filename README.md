# MV-LEAP
Multi-View LEArning-based data Proliferator (MV-LEAP) for boosting classification using highly imbalanced classes, created by by Olfa Graa.

Please contact olfa.graa@gmail.com for inquiries. Thanks.

![MV-LEAP pipeline](http://basira-lab.com/mvleap_0/)

# Introduction

This work has been published in the Journal of Neuroscience Methods 2019. MV-LEAP is a framework for boosting the classification of imbalanced multi-view data. MV-LEAP comprises two key steps addressing two major machine learning problems in classification tasks:
Issue 1: Training data imbalance.  
Proposed solution: manifold learning-based proliferator, which enables to generate synthetic data for each view, is proposed to handle imbalanced data.
Issue 2: Heterogeneity of the input multi-view data to learn from. 
Proposed solution: a multi-view manifold data alignment leveraging tensor canonical correlation analysis is proposed to map all original (i.e., ground truth) and proliferated (i.e., synthesized) views into a shared subspace where their distributions are aligned for the target classification task.
More details can be found at: https://www.sciencedirect.com/science/article/pii/S016502701930202X) or https://www.researchgate.net/publication/334162522_Multi-View_Learning-Based_Data_Proliferator_for_Boosting_Classification_Using_Highly_Imbalanced_Classes

In this repository, we release the MV-LEAP source code trained and tested in a simulated heterogeneous multi-view dataset drawn from 4 Gaussian distributions as shown below:


![Simulated heterogeneous multi-view dataset](http://basira-lab.com/mvleap_1/)

The classification results by comparison methods and MV-LEAP (ours) are displayed below:

![Demo classification results](http://basira-lab.com/mvleap_2/)

# Installation

This framework was developed on Matlab R2018a. It uses the following Matlab packages:
tensor_toolbox_2.6
SMOTE
SIMLR
PCA
TCCA

You also need to use the packages in the bib folder included in the following repositories: `bib` and `bib/src/`.
To use the bib codes, you need to load the bib path in the MV-LEAP code. Use the function: addpath('bib/')

*** Synthetic Minority Over-sampling TEchnique (SMOTE):
Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip.: SMOTE: synthetic minority over-sampling technique. [https://arxiv.org/abs/1106.1813] (2002) [https://github.com/kedarps/MATLAB-SMOTE]

*** Single‐cell Interpretation via Multi‐kernel LeaRning (SIMLR):
Wang, B., Ramazzotti, D., De Sano, L., Zhu, J., Pierson, E., Batzoglou, S.: SIMLR: a tool for large-scale single-cell analysis by multi-kernel learning. [https://www.biorxiv.org/content/10.1101/052225v3] (2017) [https://github.com/BatzoglouLabSU/SIMLR].

*** Tensor Canonical Correlation Analysis (TCCA):
Luo, Y., Tao, D., Ramamohanarao, K., Xu, C., Wen, Y.:Tensor canonical correlation analysis for multi-view dimension reduction. [https://arxiv.org/abs/1502.02330] (2015)

*** Tensor Toolbox
Kolda, Tamara G., and Bader, Brett W. MATLAB Tensor Toolbox. Computer software. Vers. 00. USDOE. 3 Aug. 2006. Web. [https://www.tensortoolbox.org] (2016) [https://gitlab.com/tensors/tensor_toolbox]

# Data
In order to use our framework, you need to provide: 
1) A cell arrays (X) including the data matrices, where each cell represents a data view. Each data matrix should be of size ns × nf (ns = number of samples and nf = number of features), where each row represents a sample, and each column represents a feature. 
2) A label vector (Y) storing the labels for all samples, where **0 represents the majority class label** and **1 represents the minority class label.**

The input data should have (.mat) format and needs to be loaded using the matlab load function. 

# Run the code
To test our code, run the demo: MV_LEAP_demo.m

A brief description of the script files: 
MV_learning.m: includes the main code of MV-LEAP.
mv_leap_figures: plots the results.
MV_LEAP_func: runs MV-LEAP functions.
Smote_func: includes data proliferation code.


# License
Our code is released under MIT License (see LICENSE file for details).

# Please cite the following paper when using MV-LEAP

@article{graa2019multi,<br/>
  title={Multi-view learning-based data proliferator for boosting classification using highly imbalanced classes},<br/>
  author={Graa, Olfa and Rekik, Islem},<br/>
  journal={Journal of neuroscience methods},<br/>
  volume={327},<br/>
  pages={108344},<br/>
  year={2019},<br/>
  }


Paper link on ResearchGate: https://www.researchgate.net/publication/334162522_Multi-View_Learning-Based_Data_Proliferator_for_Boosting_Classification_Using_Highly_Imbalanced_Classes
