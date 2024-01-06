function [accuracy, predicted_label] = test_nn(data, label, W1, W2)

num_classes = 10;

X = [ones(size(data, 1), 1) data];

predicted_label = predict_nn(X, W1, W2, num_classes);

accuracy = mean(double(predicted_label-1 == label)) * 100;

end