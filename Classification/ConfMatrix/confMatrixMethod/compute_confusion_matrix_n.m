function [confusion_matrix_n]=compute_confusion_matrix_n(predict_label,num_in_class)
%  Matlab code for computing and visulization of confusion matrix for
%  multi-classification problem
%
%  Author�� Page( ا��)
%           Blog: www.shamoxia.com;
%           QQ:379115886;
%           Email: peegeelee@gmail.com
%  Date:    Dec. 2010
%
% 
%num_in_class ÿ���������������
%name_class ÿ�������
%
%eg.(for UCM)
%predict_label    %%������
%num_in_class = ones(1, 21)*20   %%������
%name_class = {'lake', 'park'....}    %%ע��cell����


num_class=length(num_in_class);

confusion_matrix_n=zeros(num_class,num_class);

for ci=1:num_class
    for cj=1:num_class
        c_start=sum(num_in_class(1:(ci-1)))+1;
        c_end=sum(num_in_class(1:ci));
        confusion_matrix_n(ci,cj)=size(find(predict_label(c_start:c_end)==cj),1);
    end
end

% draw_cm(confusion_matrix,name_class,num_class);

end