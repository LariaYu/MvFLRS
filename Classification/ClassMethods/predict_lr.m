function [labels] = predict_lr(X, W, num_classes)

predicted_target = calc_y_lr(X, W, num_classes);

[max_labels, labels] = max(predicted_target, [], 2);

end