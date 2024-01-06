clear; close all; clc;
%参数设置
dataNames={'UCMerced'}; 

%% 数据处理
load '..\ClassResults\Md_Cotrained_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MSE_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvDA_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvDAvc_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_MvFLRS_UCMerced_Classifier_PR.mat';
load '..\ClassResults\Md_SMSE_UCMerced_Classifier_PR.mat';
OA={Multi_MvFLRS_UCMerced_PR,Multi_Cotrained_UCMerced_PR,Multi_MSE_UCMerced_PR,Multi_MvDA_UCMerced_PR,Multi_MvDAvc_UCMerced_PR,Multi_SMSE_UCMerced_PR};
lens=length(OA);
MethodNames={'MvFLRS','Cotrained','MSE','MvDA','MvDAvc','S-MSE'};
file_id=fopen('Md_UCMerced_latex.txt','w');
for i=1:lens
    tm=OA{i}.meanV;
    tmax=OA{i}.maxV;
    leng=length(tmax);
    fprintf(file_id,'%s%s%s\r\n','\multirow{2}{*}{',MethodNames{i},'}');
    fprintf(file_id,'%s%s','&Mean');
    for k=1:leng
    fprintf(file_id,'%s%4.2f',' & ',tm(k)); 
    end
    fprintf(file_id,'%s\r\n',' \\');
    fprintf(file_id,'%s%s','&Max');
    for j=1:leng
    fprintf(file_id,'%s%4.2f',' & ',tmax(j)); 
     end
    fprintf(file_id,'%s\r\n',' \\');
    fprintf(file_id,'%s\r\n',' \\');
end