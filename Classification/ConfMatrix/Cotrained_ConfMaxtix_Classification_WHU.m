%% Initialization
clear ; close all; clc
%% Load Data
load('..\..\..\..\ArticalCode\Co-training approach\MyCotrain\MvFLRS\SLCGFeatures\Cotrained_WHU.mat');
globalfeatureall=CotrainH;
Datanum='Cotrained_SLCG_WHU';
%% load train data and test data
load('..\..\..\..\SingView_Method\DatasetFeaMethods\SIRI-WHU_earth\ClassifierData\Train_Data_WHU_Labels.mat');
addpath('..\ClassMethods');
addpath('.\confMatrixMethod');
ClassData=[];
ClassData.training_data=globalfeatureall(RcnData.R_train_all',:);
ClassData.training_label=RcnData.Labels(RcnData.R_train_all)';
ClassData.test_label=RcnData.Labels(RcnData.R_test_all)';
ClassData.test_data=globalfeatureall(RcnData.R_test_all',:);
training_data=double(ClassData.training_data);
training_label=ClassData.training_label;
test_data=double(ClassData.test_data);
test_label=ClassData.test_label;
linmodel = svmtrain(training_label, training_data,'-t 1 -g 1 -r 1 -d 3');
[YPredicted,accuracy_svm,dec_value] = svmpredict(test_label,test_data,linmodel) ;
% classname=unique(test_label);
classname=RcnData.CategoryName';
classname{4}='idle-land';
num_test_class=0.4*(RcnData.imNumOfClass);
% clear globalfeatureall ClassData training_data test_data RcnData MvFLRS linmodel training_label mv_param dec_value;

[confusion_matrix]=compute_confusion_matrix(YPredicted,num_test_class,classname);
num_class=length(classname);
[confusion_matrix_n]=compute_confusion_matrix_n(YPredicted,num_test_class);
kappa = compute_kappa_coefficient(confusion_matrix_n, num_class, num_test_class);

cfname=sprintf('%s%s%s','.\confMatrix\',Datanum,'_confM.mat');
save (cfname, 'confusion_matrix','YPredicted','kappa','accuracy_svm');

strpr1=sprintf('%s%s%s','.\confMatrix\',Datanum,'_confM.fig');
saveas(gcf,strpr1);

% plotconfusion(test_label,YPredicted);
% % 
% %{ 
% LIBSVM训练时可以选择的参数很多，包括：
% 
%     -s svm类型：SVM设置类型（默认0)
%     　　　　0 — C-SVC； 1 –v-SVC； 2 – 一类SVM； 3 — e-SVR； 4 — v-SVR
%     -t 核函数类型：核函数设置类型（默认2）
%     　　　　0 – 线性核函数：u’v
%     　　　　1 – 多项式核函数：（r*u’v + coef0)^degree
%     　　　　2 – RBF(径向基)核函数：exp(-r|u-v|^2）
%     　　　　3 – sigmoid核函数：tanh(r*u’v + coef0)
%     -d degree：核函数中的degree设置（针对多项式核函数）（默认3）
%     -g r(gamma）：核函数中的gamma函数设置（针对多项式/rbf/sigmoid核函数）（默认1/k，k为总类别数)
%     -r coef0：核函数中的coef0设置（针对多项式/sigmoid核函数）（（默认0)
%     -c cost：设置C-SVC，e -SVR和v-SVR的参数（损失函数）（默认1）
%     -n nu：设置v-SVC，一类SVM和v- SVR的参数（默认0.5）
%     -p p：设置e -SVR 中损失函数p的值（默认0.1）
%     -m cachesize：设置cache内存大小，以MB为单位（默认40）
%     -e eps：设置允许的终止判据（默认0.001）
%     -h shrinking：是否使用启发式，0或1（默认1）
%     -wi weight：设置第几类的参数C为weight*C (C-SVC中的C) （默认1）
%     -v n: n-fold交互检验模式，n为fold的个数，必须大于等于2
% 
% %}
