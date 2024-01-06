%% Initialization
clear ; close all; clc

%% Load Data
%%%%%%% Train
load training_data.mat;
load training_label.mat;
load test_data.mat;
load test_label.mat;
% [weight] = train_lr(training_data, training_label);
% %%%%%%% Test

% [accuracy_lr, y_lr] = test_lr(test_data, test_label, weight);
% %%%%%%% Output
% dlmwrite('classes_lr.txt', y_lr);
% save 'classes_lr.mat' y_lr;
% %%%%%%% Train
% [weight1, weight2] = train_nn(training_data, training_label);
% %%%%%%% Test
% [accuracy_nn, y_nn] = test_nn(test_data, test_label, weight1, weight2);
% %%%%%%% Output
% dlmwrite('classes_nn.txt', y_nn);
% save 'classes_nn.mat' y_nn;
%%%%%%%%%%%%%%%%%%% Neural Networks %%%%%%%%%%%%%%%%%%%%
%linear kernel

linmodel = svmtrain(training_label, training_data, '-t 0');

[y_svm_train , accuracy_svm_train , prob_estimates_train] = svmpredict(training_label, training_data,linmodel );

[y_svm , accuracy_svm , prob_estimates] = svmpredict(test_label,test_data,linmodel ) ;