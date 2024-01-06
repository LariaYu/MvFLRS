function [Z, y] = calc_y_nn(X, weight1, weight2, num_classes)

a1 = X;
Z2 = a1*weight1;
a2 = sigmoid(Z2);
a2 = [ones(size(a2, 1), 1) a2];
Z3 = a2*weight2';
y = sigmoid(Z3);

Z = a2;

% Z = double(sigmoid(X*weight1));
% Z = [ones(size(Z, 1), 1) Z];
% exp_y = exp(Z*weight2');
% sum_exp_y = sum(exp_y, 2);
% rep_exp_y = repmat(sum_exp_y, 1, num_classes);
% y = exp_y./rep_exp_y;

end