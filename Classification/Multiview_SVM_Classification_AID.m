%% Initialization
clear ; close all; clc

%% Load Data
% GSC features extracted from each of the image: each image is represented 
% by a 512-bit vector, the rst 192 are G (gradient), the next 192 are S 
% (structural) and the last 128 are C (concavity).
addpath('.\ClassMethods');
load('../../../ImageSets/Remote Sensing dataset/AID/AID_code/AIDmyCode/AID_Labels.mat');
[RcnN,~]=size(Features.R_train_all);
DataNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
MethodNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
MethodsNum=length(MethodNames);
Accuracy=zeros(RcnN,MethodsNum);
load('../MvFLRSfeatures/MvFLRS_SLCG_AID.mat');
MvFLRS=DR_HistNorm(MvFLRS);
load('..\..\..\ArticalCode\Co-training approach\MyCotrain\MvFLRS\SLCGFeatures\Cotrained_AID.mat');
load('..\..\MSE(MultiviewSpectralEmbbeding)\Features\MSE_SLCG_AID.mat');
MSE=DR_HistNorm(AID_MSE.feature);
load('..\..\..\ArticalCode\MyMvda\MvDA_GHMP\GHMP_F_MvDa\MvDA_SLCG_AID.mat');
load('..\..\..\ArticalCode\MyMvda\MvDA_GHMP\GHMP_F_MvDa\MvDAvc_SLCG_AID.mat');
load('..\..\S-MSE(Multi_view Spectral embedding)\SLCG\SMSE_Feature\SMSE_SLCG_AID.mat');
SMSE=DR_HistNorm(SMSE);

FeaturesAll={MvFLRS,CotrainHn,MSE,MvDA,MvDAvc,SMSE};
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

Multi_MvFLRS_AID_PR.AlNames=DataNames;
Multi_MvFLRS_AID_PR.MethodNames=MethodNames;
Multi_MvFLRS_AID_PR.Accuracy=Accuracy;
Multi_MvFLRS_AID_PR.R_train_all=Features.R_train_all;
Multi_MvFLRS_AID_PR.R_test_all=Features.R_test_all;
save('.\ClassifierResult\Multi_AID_Classifier_PR.mat','Multi_MvFLRS_AID_PR');


% % 
function [accuracy_svm]=SVMClassifier(ClassData)
        training_data=double(ClassData.training_data);
        training_label=ClassData.training_label;
        test_data=double(ClassData.test_data);
        test_label=ClassData.test_label;
         linmodel = svmtrain(training_label, training_data,'-t 1 -g 1 -r 1 -d 3');
         [~,accuracy_svm,~] = svmpredict(test_label,test_data,linmodel) ;
end

