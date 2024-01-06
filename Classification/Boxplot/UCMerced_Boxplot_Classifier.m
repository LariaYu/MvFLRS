
clear;clc;
strname='UCMercedRcn';
load '..\ClassifierResult\UCMerced_Classifier_PR.mat';
FeaturesPR=MvFLRS_UCMerced_PR;
boxplot(FeaturesPR.Accuracy,FeaturesPR.MethodNames);
title('The results of the UCMerced Classification');
xlabel('Methods');
ylabel('Accuracy');
strpr1=sprintf('%s%s%s','.\FiguresAndDatas\',strname,'PR.fig');
saveas(gcf,strpr1);
RcnPR.mean=mean(FeaturesPR.Accuracy);
RcnPR.Max=max(FeaturesPR.Accuracy);
RcnPR.MethodsName=FeaturesPR.MethodNames;
save('.\FiguresAndDatas\UCMerced_RcnPR.mat','RcnPR');