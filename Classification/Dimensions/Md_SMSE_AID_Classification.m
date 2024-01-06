%% Initialization
clear ; close all; clc

%% Load Data
% GSC features extracted from each of the image: each image is represented 
% by a 512-bit vector, the rst 192 are G (gradient), the next 192 are S 
% (structural) and the last 128 are C (concavity).
addpath('..\ClassMethods');

load('../../../../ImageSets/Remote Sensing dataset/AID/AID_code/AIDmyCode/AID_Labels.mat');
[RcnN,~]=size(Features.R_train_all);
load('..\..\..\S-MSE(Multi_view Spectral embedding)\SLCG\SMSE_Feature\DSMSE_SLCG_AID.mat');
DimsNum=length(SMSE);
Accuracy=zeros(RcnN,DimsNum);
Dims=cell(1,DimsNum);


for i=1:DimsNum
    ClassDataC=[];
    globalfeatureall=DR_HistNorm(SMSE{i});
    [~,Dims{i}]=size(globalfeatureall);
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
% 
Multi_SMSE_AID_PR.DimsNum=Dims;
Multi_SMSE_AID_PR.MethodName='SMSE';
Multi_SMSE_AID_PR.Accuracy=Accuracy;
Multi_SMSE_AID_PR.R_train_all=Features.R_train_all;
Multi_SMSE_AID_PR.R_test_all=Features.R_test_all;
Multi_SMSE_AID_PR.meanV=mean(Accuracy);
Multi_SMSE_AID_PR.maxV=max(Accuracy);
save('.\ClassResults\Md_SMSE_AID_Classifier_PR.mat','Multi_SMSE_AID_PR');
% 
% 
% % % 
function [accuracy_svm]=SVMClassifier(ClassData)
        training_data=double(ClassData.training_data);
        training_label=ClassData.training_label;
        test_data=double(ClassData.test_data);
        test_label=ClassData.test_label;
         linmodel = svmtrain(training_label, training_data,'-t 1 -g 1 -r 1 -d 3');
         [~,accuracy_svm,~] = svmpredict(test_label,test_data,linmodel) ;
end
% 
