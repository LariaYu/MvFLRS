clear; close all; clc;
%参数设置
dataNames={'UCMerced'}; 
strname='UCMerced_Dim_';
%% 数据处理
load '..\ClassResults\Md_Cotrained_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MSE_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvDA_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvDAvc_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvFLRS_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_SMSE_UCMerced_Classifier_PR.mat';
OA={Multi_MvFLRS_UCMerced_PR,Multi_Cotrained_UCMerced_PR,Multi_MSE_UCMerced_PR,Multi_MvDA_UCMerced_PR,Multi_MvDAvc_UCMerced_PR,Multi_SMSE_UCMerced_PR};
lens=length(OA);
LineStyles={'sr-','<k-.','oc--','+g--','db--','*m--'};
MethodNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
hold on;
for i=1:lens
    plot(OA{i}.meanV, LineStyles{i});
end
   % 线条标注  
    xlabel('Dimension');
    ylabel('Mean of overall accuracy');
    legend(MethodNames,'Location','Best');
    legend('boxoff');
    set(gca,'XTick',1:1:6); 
    set(gca,'XTickLabel',{'10','20','30','40','50','60'}); 
 hold off;   
strpr1=sprintf('%s%s%s','.\DaccFigure\',strname,'PR.fig');
saveas(gcf,strpr1);
