function [Weight1, Weight2] = train_nn(data, label,classnums)

[num_inputs, num_features]= size(data);
num_hidden = 70;
num_classes = classnums;
learning_rate = 0.000001;
num_iterations = 700;

%%%% generating 10 different classes of binary outputs
T = zeros(num_inputs, num_classes);
for i = 1:num_classes
    T(:, i) = (label==i-1);  
end

X = [ones(num_inputs, 1) data];
Weight1 = rand(num_features+1, num_hidden)*0.2-0.1;
Weight2 = rand(num_classes, num_hidden+1)*0.5-0.25;

[Z, y] = calc_y_nn(X, Weight1, Weight2, num_classes);

min_error = 10000;
Weight1_opt = Weight1;
Weight2_opt = Weight2;


for i= 1:num_iterations

    %%%%% Calculate Error through Back Propagation
    delta_k = (y - T);
    delta_j = (1-Z.^2).*(delta_k*Weight2);

    gradient1 = X' * delta_j;
    gradient2 = delta_k' * Z;
        
    Weight1 = Weight1 - learning_rate * gradient1(:, 2:end);
    Weight2 = Weight2 - learning_rate * gradient2;
    
    [Z, y] = calc_y_nn(X, Weight1, Weight2, num_classes);
    error = -(1/num_inputs) * sum( sum(T .* log(y) + (1-T).* log(1-y)));
    
    %%% Capture Min Error
    if error < min_error
       min_error = error;
       Weight1_opt = Weight1;
       Weight2_opt = Weight2;
       opt_iteration = i;
    end
    
end  

Weight1 = Weight1_opt;
Weight2 = Weight2_opt;

end