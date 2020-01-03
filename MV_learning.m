function [Kfold_results, results, original_test_labels, predicted_test_labels] = MV_learning(X_init, labels, n_views, k_fold, SMOTE_param, TCCA_param)
% MV-LEAP performs supervised brain network analysis based on 
% imbalanced multi-view data

% INPUT
% X: cell arrays, contains the data matrices such as each cell defines a
% view
% Y: label vector of {0,1}, where 0 representes the majority class and 1
% representas the minority class
% TCCA_Dim: the rank of the tensor for the canonical polyadic decomposition (CP-ALS)
% TCCA_Epsilon: the regularization trade-off factor
% smote_ML: is a boolean, 0 for standard SMOTE, 1 for SMOTE using manifold
% learning
%
% OUTPUT
% Kfold_results : is a table containing the statictic results for k-fold
% test.
% results : is a table containing the statistic results : accuracy,
% sensitivity, specificity, precision, f-measure and g-mean.
% original_test_labels : is a vector of the origial labels of the test
% samples.
% predicted_test_labels : is a vector of the predicted labels of the test
% samples.
%
% Example: see MV_LEAP_demo.m
%
% Reference:
% Olfa GRAA, Islem REKIK. 
% Multi-View Learning-Based Data Proliferator for Boosting Classification Using Highly Imbalanced Classes.
% Journal of Neuroscience Methods, July 2019.
%
% Dependency:
% Matlab tensor toolbox v 2.6
% Brett W. Bader, Tamara G. Kolda and others
% http://www.sandia.gov/~tgkolda/TensorToolbox
%%
rng('default');
rng(1);

%flags
flag.smote_ML = SMOTE_param.smote_ML;
flag.TCCA = TCCA_param.TCCA;

train_data_final = [];
test_data_final = [];
train_labels_final_all = [];
    
c = cvpartition(size(labels,1),'KFold',k_fold);
    
original_test_labels = []; 
predicted_test_labels = [];
    
% initilaization acc variance
Acc = [];
Sen = [];
Sp = [];
Pr = [];
f_m = [];
g_m = [];
B_Acc = [];
            
for i = 1:c.NumTestSets % K-fold CV loops
    trainIndex = c.training(i);
    testIndex = c.test(i);  

    testLabels = [];
    trainLabels = labels(trainIndex);
    testLabels = labels(testIndex);

    trainData_init = {};
    testData_init = {};
    for index=1:n_views 
        trainData_init{1,index} = X_init{1,index}(trainIndex,:);
        testData_init{1,index} = X_init{1,index}(testIndex,:);
    end
    
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SMOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp("TEST "+i+": START SMOTE ");
    
    train_data_smote = {};
    train_labels_smote = {};
                 
    for i_view = 1:n_views
        [train_data_smote{1,i_view}, train_labels_smote{1,i_view}] = Smote_func(trainData_init{i_view}, trainLabels, SMOTE_param);
    end
    
    trainData = train_data_smote;
    testData = testData_init;
    
    train_labels_final = train_labels_smote{1,1}; % all train_labels_smote are equals
    train_data_final = [];
    test_data_final = [];
    for i_view = 1:n_views
        train_data_final = [train_data_final, train_data_smote{1,i_view}];
        test_data_final = [test_data_final, testData_init{1,i_view}];
    end

    disp("TEST "+i+": END SMOTE");                      
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%             
    disp("TEST "+i+": START PCA");
    % PCA initialisation
    train_data_final = [];
    test_data_final = [];
    n_pca = 100; 

    test_coeff = {};
    test_score = {};
    test_latent = {};
    test_tsquared = {};
    test_explained = {};
    test_mu = {};
    
    train_coeff = {};
    train_score = {};
    train_latent = {};
    train_tsquared = {};
    train_explained = {};
    train_mu = {};
    
    for i_view = 1:n_views
        [test_coeff{i_view}, test_score{i_view}, test_latent{i_view}, test_tsquared{i_view}, test_explained{i_view}, test_mu{i_view}] = pca(testData_init{i_view},'NumComponents', n_pca);
        [train_coeff{i_view}, train_score{i_view}, train_latent{i_view}, train_tsquared{i_view}, train_explained{i_view}, train_mu{i_view}] = pca(trainData{i_view}, 'NumComponents', size(test_score{i_view},2));
    end

    trainData = {};
    testData = {};
    trainData = train_score;
    testData = test_score;

    train_data_final = [];
    test_data_final = [];
    for i_view = 1:n_views
        train_data_final = [train_data_final, trainData{1,i_view}];
        test_data_final = [test_data_final, testData{1,i_view}];
    end
        
    disp("TEST "+i+": END PCA");    
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Tensor CCA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    if flag.TCCA
    
        disp("TEST "+i+": START TCCA");    

        set.nbV = size(trainData,2);

        % Set dimensionality of TCCA         
        para.rDim = TCCA_param.TCCA_Dim;
        % Set epsilon parameter
        para.epsilon = TCCA_param.TCCA_Epsilon;

        [var_mats, cov_ten] = var_cov_ten_calculation(trainData);
        [trainH, trainZ] = TCCA(trainData, var_mats, cov_ten, set, para);

        [testH, testZ] = TCCA(testData, var_mats, cov_ten, set, para);

        trainMat = [];
        testMat = [];
        for z3 = 1:size(trainZ,1)
            trainMat = [trainMat trainZ{z3}];
            testMat = [testMat testZ{z3}];
        end

        train_data_final = [];
        test_data_final = [];
        test_data_final = testMat;
        train_data_final = trainMat;

        disp("TEST "+i+": END TCCA"); 
    end
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SVM MV-LEAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SVMModel = fitcsvm(train_data_final,train_labels_final,'Standardize',true,'KernelFunction','linear',...
                    'KernelScale','auto');
    predict_label = [];
    score = [];
    [predict_label,score] = predict(SVMModel,test_data_final);

    predicted_test_labels = [predicted_test_labels ; predict_label];
    original_test_labels = [original_test_labels ; testLabels];        
%%    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Confusion matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CM_in = [];
    CM_in = confusionmat(testLabels,predict_label); 
    True_Positive_in = CM_in(1,1);
    True_Negative_in = CM_in(2,2);
    False_Positive_in = CM_in(2,1);
    False_Negative_in = CM_in(1,2);
         
    Accuracy_in = (True_Positive_in + True_Negative_in)/(size(test_data_final,1)) * 100;
    Sensitivity_in = (True_Positive_in)/(True_Positive_in + False_Negative_in) * 100;
    Specificity_in = (True_Negative_in)/(True_Negative_in+ False_Positive_in) * 100;
    Precision_in = (True_Positive_in)/(True_Positive_in + False_Positive_in) * 100;
    F_measure_in = (2 * Sensitivity_in * Precision_in)/(Sensitivity_in + Precision_in);
    G_mean_in = sqrt(Sensitivity_in * Specificity_in);
         
    Acc = [Acc, Accuracy_in];
    Sen = [Sen , Sensitivity_in ];
    Sp = [Sp , Specificity_in ];
    Pr = [Pr , Precision_in ];
    f_m = [f_m , F_measure_in ];
    g_m = [g_m , G_mean_in ];
    B_Acc = [B_Acc , (Sensitivity_in + Specificity_in)/2];
         
    train_labels_final_get_all{i} = train_labels_final;
         
    train_labels_input_all{i} = trainLabels;

end


CM = [];
CM = confusionmat(original_test_labels,predicted_test_labels); 
True_Positive = CM(1,1);
True_Negative = CM(2,2);
False_Positive = CM(2,1);
False_Negative = CM(1,2);

Accuracy = (True_Positive + True_Negative)/(size(original_test_labels,1)) * 100;
Sensitivity = (True_Positive)/(True_Positive + False_Negative) * 100;
Specificity = (True_Negative)/(True_Negative+ False_Positive) * 100;
Precision = (True_Positive)/(True_Positive + False_Positive) * 100;
F_measure = (2 * Sensitivity * Precision)/(Sensitivity + Precision);
G_mean = sqrt(Sensitivity * Specificity);

Balanced_Accuracy = (Sensitivity + Specificity)/2;

%Results
kfield_final_results = [Acc ; Sen ; Sp ; Pr ; f_m ; g_m; B_Acc];
final_results = [Accuracy; Sensitivity; Specificity; Precision; F_measure; G_mean; Balanced_Accuracy];
            
train_labels_final_all = train_labels_final_get_all;
         
train_labels_input = train_labels_input_all;

rows_title = {'Acc%';'Sen%';'Sp%';'Precision%';'F_measure%';'G_mean%'; 'B_Acc'};
Kfold_results = table(kfield_final_results,'RowNames',rows_title);
results = table(final_results,'RowNames',rows_title);
