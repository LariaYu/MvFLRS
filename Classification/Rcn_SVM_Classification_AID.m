%% Initialization
clear ; close all; clc

%% Load Data
% GSC features extracted from each of the image: each image is represented 
% by a 512-bit vector, the rst 192 are G (gradient), the next 192 are S 
% (structural) and the last 128 are C (concavity).
addpath('.\ClassMethods');
load('../../../ImageSets/Remote Sensing dataset/AID/AID_code/AIDmyCode/AID_Labels.mat');
featureDir='../../../ImageSets/Remote Sensing dataset/AID/AID_code/temp_data/glofeat/AID/globalfeatureall_lowlevel_';
[RcnN,~]=size(Features.R_train_all);
DataNames={'gist','ch','lbp256','sift','MvFLRS'};
MethodNames={'Gist','CH','LBP','SIFT','MvFLRS'};
% DataNames={'gist','MvFLRS'};
% MethodNames={'Gist','MvFLRS'};
Multiview=length(DataNames);
Accuracy=zeros(RcnN,Multiview);

for i=1:Multiview-1
    ClassDataC=[];
    load(sprintf('%s%s%s',featureDir,DataNames{i},'.mat'));
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
%% MvFLRS Features
load('../MvFLRSfeatures/MvFLRS_SLCG_AID.mat');
MvFLRS=DR_HistNorm(MvFLRS);
% MvFLRS=abs(MvFLRS);
ClassDataC=[];
    for j=1:RcnN
        ClassDataC.training_data=MvFLRS(Features.R_train_all(j,:)',:);
        ClassDataC.training_label=Features.Labels(Features.R_train_all(j,:))';
        ClassDataC.test_label=Features.Labels(Features.R_test_all(j,:))';
        ClassDataC.test_data=MvFLRS(Features.R_test_all(j,:)',:);
        [Acc]=SVMClassifier(ClassDataC);
        Accuracy(j,Multiview)=Acc(1);
    end


MvFLRS_AID_PR.AlNames=DataNames;
MvFLRS_AID_PR.MethodNames=MethodNames;
MvFLRS_AID_PR.Accuracy=Accuracy;
MvFLRS_AID_PR.R_train_all=Features.R_train_all;
MvFLRS_AID_PR.R_test_all=Features.R_test_all;
save('.\ClassifierResult\AID_Classifier_PR.mat','MvFLRS_AID_PR');


% % 
function [accuracy_svm]=SVMClassifier(ClassData)
    training_data=double(ClassData.training_data);
    training_label=ClassData.training_label;
    test_data=double(ClassData.test_data);
    test_label=ClassData.test_label;
    %{  %�����������������һ��Ч�����
        %     linmodel = svmtrain(training_label, training_data, '-t 3 -c 2 -g 0.01');   %[7.45,8.225]
        %     linmodel = svmtrain(training_label, training_data, '-t 2 -c 6 -g 0.02');   %[19.2625,8.225]
        %     linmodel = svmtrain(training_label, training_data, '-t 1 -c 2 -g 0.01');   %[5.825,8.2375]
        %       linmodel = svmtrain(training_label, training_data, '-t 1 -g 0.01');   %[5.825,8.2375]  
        %          linmodel = svmtrain(training_label, training_data, '-t 1');  %[5.825,8.2375]
    %}
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
