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
