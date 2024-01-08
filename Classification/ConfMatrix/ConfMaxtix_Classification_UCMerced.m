%% Initialization
clear ; close all; clc
%% Load Data
load('../MvFLRSFeatures/MvFLRS_SLCG_UCMerced.mat');
globalfeatureall=DR_HistNorm(MvFLRS);
Datanum='SLCG_UCMerced';
%% load train data and test data
load('..\..\..\SingView_Method\DatasetFeaMethods\UCMerced\ClassifierData\RcnData_UCMerced_Labels.mat');
addpath('.\ClassMethods');
j=3;
ClassData=[];
ClassData.training_data=globalfeatureall(RcnData.R_train_all(j,:)',:);
ClassData.training_label=RcnData.Labels(RcnData.R_train_all(j,:))';
ClassData.test_label=RcnData.Labels(RcnData.R_test_all(j,:))';
ClassData.test_data=globalfeatureall(RcnData.R_test_all(j,:)',:);
training_data=double(ClassData.training_data);
training_label=ClassData.training_label;
test_data=double(ClassData.test_data);
test_label=ClassData.test_label;
linmodel = svmtrain(training_label, training_data,'-t 1 -g 1 -r 1 -d 3');
[YPredicted,accuracy_svm,dec_value] = svmpredict(test_label,test_data,linmodel) ;

plotconfusion(test_label,YPredicted);
% 
% %{ 
% LIBSVMѵ��ʱ����ѡ��Ĳ����ܶ࣬������
% 
%     -s svm���ͣ�SVM�������ͣ�Ĭ��0)
%     ��������0 �� C-SVC�� 1 �Cv-SVC�� 2 �C һ��SVM�� 3 �� e-SVR�� 4 �� v-SVR
%     -t �˺������ͣ��˺����������ͣ�Ĭ��2��
%     ��������0 �C ���Ժ˺�����u��v
%     ��������1 �C ����ʽ�˺�������r*u��v + coef0)^degree
%     ��������2 �C RBF(�����)�˺�����exp(-r|u-v|^2��
%     ��������3 �C sigmoid�˺�����tanh(r*u��v + coef0)
%     -d degree���˺����е�degree���ã���Զ���ʽ�˺�������Ĭ��3��
%     -g r(gamma�����˺����е�gamma�������ã���Զ���ʽ/rbf/sigmoid�˺�������Ĭ��1/k��kΪ�������)
%     -r coef0���˺����е�coef0���ã���Զ���ʽ/sigmoid�˺���������Ĭ��0)
%     -c cost������C-SVC��e -SVR��v-SVR�Ĳ�������ʧ��������Ĭ��1��
%     -n nu������v-SVC��һ��SVM��v- SVR�Ĳ�����Ĭ��0.5��
%     -p p������e -SVR ����ʧ����p��ֵ��Ĭ��0.1��
%     -m cachesize������cache�ڴ��С����MBΪ��λ��Ĭ��40��
%     -e eps�������������ֹ�оݣ�Ĭ��0.001��
%     -h shrinking���Ƿ�ʹ������ʽ��0��1��Ĭ��1��
%     -wi weight�����õڼ���Ĳ���CΪweight*C (C-SVC�е�C) ��Ĭ��1��
%     -v n: n-fold��������ģʽ��nΪfold�ĸ�����������ڵ���2
% 
% %}
