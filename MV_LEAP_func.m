function [Kfold_results, results, original_test_labels, predicted_test_labels] = MV_LEAP_func(X, Y, nbr_views, k_fold, SMOTE_param, TCCA_param, plot_figures)

%% Initalization

labels = Y; 
X_init = X;
n_views = size(X_init,2); %get number of views
if n_views ~= nbr_views 
    error("The number of views is not valid! PLEASE correct your input");
end

SMOTE_param.smote_ML = 1;
TCCA_param.TCCA = 1;

%% MV-LEAP function 
[Kfold_results, results, original_test_labels, predicted_test_labels] = MV_learning(X_init, labels, n_views, k_fold, SMOTE_param, TCCA_param);


%% Plot figures
if plot_figures == 1
    % SMOTE_param + ML + PCA
    SMOTE_param.smote_ML = 1;
    TCCA_param.TCCA = 0;
    [Kfield_results_SMOTE_ML_PCA, results_SMOTE_ML_PCA, original_test_labels_SMOTE_ML_PCA, predicted_test_labels_SMOTE_ML_PCA] = MV_learning(X_init, labels, n_views, k_fold, SMOTE_param, TCCA_param);
    
    
    % SMOTE_param + PCA
    SMOTE_param.smote_ML = 0;
    TCCA_param.TCCA = 0;
    [Kfield_results_SMOTE_PCA, results_SMOTE_PCA, original_test_labels_SMOTE_PCA, predicted_test_labels_SMOTE_PCA] = MV_learning(X_init, labels, n_views, k_fold, SMOTE_param, TCCA_param);
    
    mv_leap_figures(original_test_labels, predicted_test_labels, results, original_test_labels_SMOTE_ML_PCA, predicted_test_labels_SMOTE_ML_PCA, results_SMOTE_ML_PCA, original_test_labels_SMOTE_PCA, predicted_test_labels_SMOTE_PCA, results_SMOTE_PCA);
end
%%


disp("END of MVLEAP.");


end