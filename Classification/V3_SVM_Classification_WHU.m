%% Initialization
clear ; close all; clc

%% Load Data
% GSC RcnData extracted from each of the image: each image is represented 
% by a 512-bit vector, the rst 192 are G (gradient), the next 192 are S 
% (structural) and the last 128 are C (concavity).
load('../MvFLRSFeatures/MvFLRS_SLG_WHU.mat');
SLG=DR_HistNorm(MvFLRS);
load('../MvFLRSFeatures/MvFLRS_SLC_WHU.mat');
SLC=DR_HistNorm(MvFLRS);
load('../MvFLRSFeatures/MvFLRS_SCG_WHU.mat');
SCG=DR_HistNorm(MvFLRS);
load('../MvFLRSFeatures/MvFLRS_LCG_WHU.mat');
LCG=DR_HistNorm(MvFLRS);
load('../MvFLRSFeatures/MvFLRS_SLCG_WHU.mat');
MvFLRS=DR_HistNorm(MvFLRS);
Datanum='V3_WHU';
MethodNames={'SLG','SLC','SCG','LCG','MvFLRS'};
Multi_view_fetures={SLG,SLC,SCG,LCG,MvFLRS};
clear SLG SLC SCG LCG MvFLRS;
%% load train data and test data
load('..\..\..\SingView_Method\DatasetFeaMethods\SIRI-WHU_earth\ClassifierData\RcnData_WHU_Labels.mat');
addpath('.\ClassMethods'); 
[RcnN,~]=size(RcnData.R_train_all);

Multiview=length(MethodNames);
Accuracy=zeros(RcnN,Multiview);
% 
for i=1:Multiview
    ClassDataC=[];
    globalfeatureall=Multi_view_fetures{i}; 
    for j=1:RcnN
        ClassDataC.training_data=globalfeatureall(RcnData.R_train_all(j,:)',:);
        ClassDataC.training_label=RcnData.Labels(RcnData.R_train_all(j,:))';
        ClassDataC.test_label=RcnData.Labels(RcnData.R_test_all(j,:))';
        ClassDataC.test_data=globalfeatureall(RcnData.R_test_all(j,:)',:);
        [Acc]=SVMClassifier(ClassDataC);
        Accuracy(j,i)=Acc(1);
    end
end
clear globalfeatureall;
MvFLRS_WHU_PR.MethodNames=MethodNames;
MvFLRS_WHU_PR.Accuracy=Accuracy;
MvFLRS_WHU_PR.R_train_all=RcnData.R_train_all;
MvFLRS_WHU_PR.R_test_all=RcnData.R_test_all;
save('.\ClassifierResult\WHU_CV3_PR.mat','MvFLRS_WHU_PR');

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
