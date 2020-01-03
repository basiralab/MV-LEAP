function [train_data_smote, train_labels_smote] = Smote_func(train_data_in, train_label_in, SMOTE_param)

    smote_type = SMOTE_param.smote_ML;
    k_SMOTE = SMOTE_param.k_SMOTE;
    k_SIMLR = SMOTE_param.k_ML;
    
    size_class_1 = 0;
    size_class_2 = 0;
    N = 100;
    X_data = [];
    X_smote = [];
    train_labels_smote = [];
    train_data_smote = [];
        
    size_class_1 = sum(train_label_in == 1); %minority class
    size_class_2 = sum(train_label_in == 0); %majority class

    N = (size_class_2-size_class_1)*100/size_class_1; %equal classes

    X_data = train_data_in(1:size_class_1,:);
        
    if smote_type == 0
        X_smote = SMOTE(X_data, N, k_SMOTE); %traditional SMOTE
    elseif smote_type == 1
        C = 2;
        X_smote = SMOTE_ML(X_data, N, k_SIMLR, C); %SMOTE using ML
    end
        
    size_min = size(X_smote,1);
    size_all = size(train_data_in,1);

    train_labels_smote(1:size_min,1) = 1;
    train_labels_smote(size_min+1:size_min+size_class_2,1) = 0;

    size_data = size(train_labels_smote,1);

    train_data_smote = [X_smote ; train_data_in(size_class_1+1:end,:)];
end