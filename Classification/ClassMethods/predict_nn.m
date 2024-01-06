function [labels] = predict_nn(X, W1, W2, num_classes)

[~, predicted_target] = calc_y_nn(X, W1, W2, num_classes);

[max_labels, labels] = max(predicted_target, [], 2);

end