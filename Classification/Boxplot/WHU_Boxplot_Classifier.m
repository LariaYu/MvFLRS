clear;clc;
strname='WHU_Rcn';
load '..\ClassifierResult\WHU_Classifier_PR.mat';
FeaturesPR=MvFLRS_WHU_PR;
boxplot(FeaturesPR.Accuracy,FeaturesPR.MethodNames);
title('The results of the WHU Classification');
xlabel('Methods');
ylabel('Accuracy');
strpr1=sprintf('%s%s%s','.\FiguresAndDatas\',strname,'PR.fig');
saveas(gcf,strpr1);
RcnPR.mean=mean(FeaturesPR.Accuracy);
RcnPR.Max=max(FeaturesPR.Accuracy);
RcnPR.MethodsName=FeaturesPR.MethodNames;
save('.\FiguresAndDatas\WHU_RcnPR.mat','RcnPR');