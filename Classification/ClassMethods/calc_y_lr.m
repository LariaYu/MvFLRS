function [y] = calc_y_lr(X, W, num_classes)

expa = double(exp(X*W));
sum_nth_row = sum(expa, 2);
sum_nth_row_mat = repmat(sum_nth_row, 1, num_classes);
y = expa./sum_nth_row_mat;

end