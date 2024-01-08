function [ kappa ] = compute_kappa_coefficient(confusion_matrix, class_num, test_num)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
N = sum(test_num);
mat = confusion_matrix;
sum_diag = trace(mat);
sum_row = sum(mat');
sum_col = sum(mat);
kappa = (N.*sum_diag-sum(sum_row.*sum_col))/(N.^2-sum(sum_row.*sum_col));

end

