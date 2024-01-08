clear; close all; clc;
%��������
dataNames={'AID'}; 
strname='AID_Dim_';
%% ���ݴ���
load '..\ClassResults\Md_Cotrained_AID_Classifier_PR.mat';
load '..\ClassResults\Md_MSE_AID_Classifier_PR.mat';
load '..\ClassResults\Md_MvDA_AID_Classifier_PR.mat';
load '..\ClassResults\Md_MvDAvc_AID_Classifier_PR.mat';
load '..\ClassResults\Md_MvFLRS_AID_Classifier_PR.mat';
load '..\ClassResults\Md_SMSE_AID_Classifier_PR.mat';
OA={Multi_MvFLRS_AID_PR,Multi_Cotrained_AID_PR,Multi_MSE_AID_PR,Multi_MvDA_AID_PR,Multi_MvDAvc_AID_PR,Multi_SMSE_AID_PR};
lens=length(OA);
LineStyles={'sr-','<k-.','oc--','+g--','db--','*m--'};
MethodNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
hold on;
for i=1:lens
    plot(OA{i}.meanV, LineStyles{i});
end
   % ������ע  
    xlabel('Dimension');
    ylabel('Mean of overall accuracy');
    legend(MethodNames,'Location','Best');
    legend('boxoff');
    set(gca,'XTick',1:1:6); 
    set(gca,'XTickLabel',{'10','20','30','40','50','60'}); 
 hold off;   
strpr1=sprintf('%s%s%s','.\DaccFigure\',strname,'PR.fig');
saveas(gcf,strpr1);
