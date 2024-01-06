function [accuracy, predicted_label] = test_lr(data, label, W)

num_classes = size(W, 2);

X = [ones(size(data, 1), 1) data];

predicted_label = predict_lr(X, W, num_classes);

accuracy = mean(double(predicted_label-1 == label)) * 100;

end