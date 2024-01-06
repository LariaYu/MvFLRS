clear;clc;
strname='Multi_WHU_Rcn';
load '..\ClassifierResult\Multi_WHU_Classifier_PR.mat';
FeaturesPR=Multi_MvFLRS_WHU_PR;
boxplot(FeaturesPR.Accuracy,FeaturesPR.MethodNames);
title('The results of the WHU Classification');
xlabel('Methods');
ylabel('Accuracy');
strpr1=sprintf('%s%s%s','.\FiguresAndDatas\',strname,'PR.fig');
saveas(gcf,strpr1);
RcnPR.mean=mean(FeaturesPR.Accuracy);
RcnPR.Max=max(FeaturesPR.Accuracy);
RcnPR.MethodsName=FeaturesPR.MethodNames;
save('.\FiguresAndDatas\Multi_WHU_RcnPR.mat','RcnPR');