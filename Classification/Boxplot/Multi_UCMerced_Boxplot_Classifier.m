
clear;clc;
strname='Multi_UCMercedRcn';
load '..\ClassifierResult\Multi_UCMerced_Classifier_PR.mat';
FeaturesPR=Multi_MvFLRS_UCMerced_PR;
boxplot(FeaturesPR.Accuracy,FeaturesPR.MethodNames);
title('The results of the UCMerced Classification');
xlabel('Methods');
ylabel('Accuracy');
strpr1=sprintf('%s%s%s','.\FiguresAndDatas\',strname,'PR.fig');
saveas(gcf,strpr1);
RcnPR.mean=mean(FeaturesPR.Accuracy);
RcnPR.Max=max(FeaturesPR.Accuracy);
RcnPR.MethodsName=FeaturesPR.MethodNames;
save('.\FiguresAndDatas\Multi_UCMerced_RcnPR.mat','RcnPR');