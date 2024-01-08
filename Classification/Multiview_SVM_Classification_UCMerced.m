%% Initialization
clear ; close all; clc

%% Load Data
% GSC features extracted from each of the image: each image is represented 
% by a 512-bit vector, the rst 192 are G (gradient), the next 192 are S 
% (structural) and the last 128 are C (concavity).
addpath('.\ClassMethods');
%% load train data and test data
load('..\..\..\SingView_Method\DatasetFeaMethods\UCMerced\ClassifierData\RcnData_UCMerced_Labels.mat');
Features=RcnData;
[RcnN,~]=size(Features.R_train_all);
DataNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
MethodNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
MethodsNum=length(MethodNames);
Accuracy=zeros(RcnN,MethodsNum);
load('..\MvFLRSfeatures\MvFLRS_SLCG_UCMerced.mat');
MvFLRS=DR_HistNorm(MvFLRS);
load('..\..\..\ArticalCode\Co-training approach\MyCotrain\MvFLRS\SLCGFeatures\Cotrained_UCMerced.mat');
load('..\..\MSE(MultiviewSpectralEmbbeding)\Features\MSE_SLCG_UCMerced.mat');
MSE=DR_HistNorm(UCMerced_MSE.feature);
load('..\..\..\ArticalCode\MyMvda\MvDA_GHMP\GHMP_F_MvDa\MvDA_SLCG_UCMerced.mat');
load('..\..\..\ArticalCode\MyMvda\MvDA_GHMP\GHMP_F_MvDa\MvDAvc_SLCG_UCMerced.mat');
load('..\..\S-MSE(Multi_view Spectral embedding)\SLCG\SMSE_Feature\SMSE_SLCG_UCMerced.mat');
SMSE=DR_HistNorm(SMSE);

FeaturesAll={MvFLRS,CotrainH,MSE,MvDA,MvDAvc,SMSE};

for i=1:MethodsNum
    ClassDataC=[];
    globalfeatureall=FeaturesAll{i};
    for j=1:RcnN
        ClassDataC.training_data=globalfeatureall(Features.R_train_all(j,:)',:);
        ClassDataC.training_label=Features.Labels(Features.R_train_all(j,:))';
        ClassDataC.test_label=Features.Labels(Features.R_test_all(j,:))';
        ClassDataC.test_data=globalfeatureall(Features.R_test_all(j,:)',:);
        [Acc]=SVMClassifier(ClassDataC);
        Accuracy(j,i)=Acc(1);
    end
end
clear globalfeatureall;

Multi_MvFLRS_UCMerced_PR.AlNames=DataNames;
Multi_MvFLRS_UCMerced_PR.MethodNames=MethodNames;
Multi_MvFLRS_UCMerced_PR.Accuracy=Accuracy;
Multi_MvFLRS_UCMerced_PR.R_train_all=Features.R_train_all;
Multi_MvFLRS_UCMerced_PR.R_test_all=Features.R_test_all;
save('.\ClassifierResult\Multi_UCMerced_Classifier_PR.mat','Multi_MvFLRS_UCMerced_PR');


% % 
function [accuracy_svm]=SVMClassifier(ClassData)
        training_data=double(ClassData.training_data);
        training_label=ClassData.training_label;
        test_data=double(ClassData.test_data);
        test_label=ClassData.test_label;
        linmodel = svmtrain(training_label, training_data,'-t 1 -g 1 -r 1 -d 3');
        [~,accuracy_svm,~] = svmpredict(test_label,test_data,linmodel) ;
end

%{ 
LIBSVMѵ��ʱ����ѡ��Ĳ����ܶ࣬������

    -s svm���ͣ�SVM�������ͣ�Ĭ��0)
    ��������0 �� C-SVC�� 1 �Cv-SVC�� 2 �C һ��SVM�� 3 �� e-SVR�� 4 �� v-SVR
    -t �˺������ͣ��˺����������ͣ�Ĭ��2��
    ��������0 �C ���Ժ˺�����u��v
    ��������1 �C ����ʽ�˺�������r*u��v + coef0)^degree
    ��������2 �C RBF(�����)�˺�����exp(-r|u-v|^2��
    ��������3 �C sigmoid�˺�����tanh(r*u��v + coef0)
    -d degree���˺����е�degree���ã���Զ���ʽ�˺�������Ĭ��3��
    -g r(gamma�����˺����е�gamma�������ã���Զ���ʽ/rbf/sigmoid�˺�������Ĭ��1/k��kΪ�������)
    -r coef0���˺����е�coef0���ã���Զ���ʽ/sigmoid�˺���������Ĭ��0)
    -c cost������C-SVC��e -SVR��v-SVR�Ĳ�������ʧ��������Ĭ��1��
    -n nu������v-SVC��һ��SVM��v- SVR�Ĳ�����Ĭ��0.5��
    -p p������e -SVR ����ʧ����p��ֵ��Ĭ��0.1��
    -m cachesize������cache�ڴ��С����MBΪ��λ��Ĭ��40��
    -e eps�������������ֹ�оݣ�Ĭ��0.001��
    -h shrinking���Ƿ�ʹ������ʽ��0��1��Ĭ��1��
    -wi weight�����õڼ���Ĳ���CΪweight*C (C-SVC�е�C) ��Ĭ��1��
    -v n: n-fold��������ģʽ��nΪfold�ĸ�����������ڵ���2

%}
