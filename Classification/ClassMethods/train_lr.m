function [W] = train_lr(data, label,classnums)

num_classes =classnums;
num_features = size(data, 2);
num_inputs = size(label, 1);
learning_rate = 0.00003;
num_iterations = 5000;

%%%% generating 10 different classes of binary outputs
T = zeros(num_inputs, num_classes);
for i = 1:num_classes
    T(:, i) = (label==i-1);  
end


X = [ones(num_inputs, 1) data];
W = zeros(num_features+1, num_classes);

y = calc_y_lr(X, W, num_classes);
min_error = 10^4;
W_opt = W;

%%%%%%% calculate weights using gradient descent %%%%%%%%
for i = 1:num_iterations
    gradient = X'*(y-T);
    W_new = W - gradient*learning_rate;
    W = W_new;
    
    y = calc_y_lr(X, W, num_classes);
    
    %%%% error
    error = abs(sum(sum(T.*log(y))));
    
    if error<min_error;
       min_error = error;
       W_opt = W;
    end
end
   
W = W_opt;

end