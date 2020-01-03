% MV-LEAP performs supervised brain network analysis based on imbalanced multi-view data
% Example: MV_LEAP_demo.m

% Paper reference:
% Olfa Graa and Islem Rekik. 
% Multi-View Learning-Based Data Proliferator for Boosting Classification Using Highly Imbalanced Classes.
% Journal of Neuroscience Methods, July 2019.
% ResearchGate Link: https://www.researchgate.net/publication/334162522_Multi-View_Learning-Based_Data_Proliferator_for_Boosting_Classification_Using_Highly_Imbalanced_Classes
% Website: https://www.sciencedirect.com/science/article/pii/S016502701930202X
% For more about our work, please check: http://basira-lab.com/ 
% YouTube channel: https://www.youtube.com/watch?v=9HbLxNef2t8&list=PLug43ldmRSo0bX8cOSuMWinXs-xem9q4o

% Dependency:
% Matlab tensor toolbox v 2.6
% Brett W. Bader, Tamara G. Kolda and others
% http://www.sandia.gov/~tgkolda/TensorToolbox

%%
%%

clear all
close all
clc;

% Load paths
addpath('Bib');
addpath('Bib/src');
addpath('tensor_toolbox-master'); %download it from http://www.sandia.gov/~tgkolda/TensorToolbox


%% Simulate data from four Gaussian distributions (4 views)
%%

nbr_views = 4; % number of views
k_fold = 5; % number of folds for k-fold cross validation
plot_figures = 1; % flag for figures, 1: plot the figures, 2: do not plot figures.


n = 100; % number of samples
d = 200; % number of features
% 'features_i' denotes the matrix of the ith data view of size n x d (n: number of samples, d: number of features)


% simulate multi-view data (in this demo, we consider 4 views)
% simulate features_1 drawn from a normal distribution N(0,1)
mu_1 = 0;
vr_1 = 1;
features_1 = mu_1 + sqrt(vr_1)*randn(n,d);

% simulate features_2 drawn from a normal distribution N(1,0.5)
mu_2 = 1;
vr_2 = 0.5;
features_2 = mu_2 + sqrt(vr_2)*randn(n,d);

% simulate features_3 drawn from a normal distribution N(2,1.5)
mu_3 = 2;
vr_3 = 1.5;
features_3 = mu_3 + sqrt(vr_3)*randn(n,d);

% simulate features_4 drawn from a normal distribution N(3,2)
mu_4 = 3;
vr_4 = 2;
features_4 = mu_4 + sqrt(vr_4)*randn(n,d);

% plot distributions 
h1 = histfit(features_1(:),10,'normal')
h1(1).FaceColor = [.8 .8 1];
h1(2).Color = [.8 .8 1];
set(h1(1),'FaceAlpha',.25);

hold on
h2 = histfit(features_2(:),10,'normal')
h2(1).FaceColor = [0.6350 0.0780 0.1840];
h2(2).Color = [0.6350 0.0780 0.1840];
set(h2(1),'FaceAlpha',.25);


hold on
h3 = histfit(features_3(:),10,'normal')
h3(1).FaceColor = [0.3010 0.7450 0.9330];
h3(2).Color = [0.3010 0.7450 0.9330];
set(h3(1),'FaceAlpha',.25);

hold on
h4 = histfit(features_4(:),10,'normal')
h4(1).FaceColor = [0.5 1 0.1];
h4(2).Color = [0.5 1 0.1];
set(h4(1),'FaceAlpha',.25);
title('Heterogeneous and entangled distribution of a simulated multiview dataset (4 views)')


%% simulate labels (make sure you assign '1' to minority class samples and '0' to majority class samples)
%%
labels(1:floor(n/4))=1; % minority class is set to 1
labels(floor(n/4+1:n))=0; % majority class is set to 0

labels = labels';

X = {features_1, features_2, features_3, features_4}; 
Y = labels; %labels vector of size n x 1 (storing the labels of all samples)


%% Hyperparameters setting
%%

SMOTE_param.k_SMOTE = 5; % set the number of the nearest neighbor for SMOTE_param
SMOTE_param.k_ML = 5; % Set the number of the nearest neighbor for ML

TCCA_param.TCCA_Dim = 30; % set the rank of the tensor for the canonical polyadic decomposition (CP-ALS)
TCCA_param.TCCA_Epsilon = 0.5; % set the regularization trade-off factor (Epsilon) 

[Kfold_results, results, original_test_labels, predicted_test_labels] = MV_LEAP_func(X, Y, nbr_views, k_fold, SMOTE_param, TCCA_param, plot_figures);

disp("If you use MV-LEAP, please cite our paper: Graa et al. (2019), Multi-View Learning-Based Data Proliferator for Boosting Classification Using Highly Imbalanced Classes. Journal of Neuroscience Methods.");
