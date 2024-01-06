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
LIBSVM训练时可以选择的参数很多，包括：

    -s svm类型：SVM设置类型（默认0)
    　　　　0 — C-SVC； 1 –v-SVC； 2 – 一类SVM； 3 — e-SVR； 4 — v-SVR
    -t 核函数类型：核函数设置类型（默认2）
    　　　　0 – 线性核函数：u’v
    　　　　1 – 多项式核函数：（r*u’v + coef0)^degree
    　　　　2 – RBF(径向基)核函数：exp(-r|u-v|^2）
    　　　　3 – sigmoid核函数：tanh(r*u’v + coef0)
    -d degree：核函数中的degree设置（针对多项式核函数）（默认3）
    -g r(gamma）：核函数中的gamma函数设置（针对多项式/rbf/sigmoid核函数）（默认1/k，k为总类别数)
    -r coef0：核函数中的coef0设置（针对多项式/sigmoid核函数）（（默认0)
    -c cost：设置C-SVC，e -SVR和v-SVR的参数（损失函数）（默认1）
    -n nu：设置v-SVC，一类SVM和v- SVR的参数（默认0.5）
    -p p：设置e -SVR 中损失函数p的值（默认0.1）
    -m cachesize：设置cache内存大小，以MB为单位（默认40）
    -e eps：设置允许的终止判据（默认0.001）
    -h shrinking：是否使用启发式，0或1（默认1）
    -wi weight：设置第几类的参数C为weight*C (C-SVC中的C) （默认1）
    -v n: n-fold交互检验模式，n为fold的个数，必须大于等于2

%}
