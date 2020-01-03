function mv_leap_figures(original_test_labels, predicted_test_labels, results, original_test_labels_SMOTE_ML_PCA, predicted_test_labels_SMOTE_ML_PCA, results_SMOTE_ML_PCA, original_test_labels_SMOTE_PCA, predicted_test_labels_SMOTE_PCA, results_SMOTE_PCA)

[x1,y1,~,auc1] = perfcurve(original_test_labels(:,1),predicted_test_labels(:,1),1);
[x2,y2,~,auc2] = perfcurve(original_test_labels_SMOTE_ML_PCA(:,1), predicted_test_labels_SMOTE_ML_PCA(:,1),1);
[x3,y3,~,auc3] = perfcurve(original_test_labels_SMOTE_PCA(:,1),predicted_test_labels_SMOTE_PCA(:,1),1);
    

figure,
hold on,
colormap winter
c = categorical({'SMOTE + PCA', 'SMOTE + ML + PCA', 'MV-LEAP (ours)'});
c = reordercats(c,{'SMOTE + PCA', 'SMOTE + ML + PCA', 'MV-LEAP (ours)'});
bar(c,[results_SMOTE_PCA{'B_Acc','final_results'}, results_SMOTE_ML_PCA{'B_Acc','final_results'}, results{'B_Acc','final_results'} ;...
    results_SMOTE_PCA{'Acc%','final_results'}, results_SMOTE_ML_PCA{'Acc%','final_results'}, results{'Acc%','final_results'} ;...
    results_SMOTE_PCA{'F_measure%','final_results'}, results_SMOTE_ML_PCA{'F_measure%','final_results'}, results{'F_measure%','final_results'}]);
title('Balanced Accuracy - Overall Accuracy - F-measure')
legend('Balanced Accuracy %', 'Overall Accuracy %', 'F-measure %')

%xlabel('1', '2', '3');
%ylabel('Balanced Accuracy %');
hold off,