clear;
clc;
close all;

% 128 * 128 = 16384
training = imageDatastore('p_dataset_26\dataset\train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

testing = imageDatastore('p_dataset_26\dataset\test', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

sizee = 16384; 
num_training = numel(training.Files);
num_testing = numel(testing.Files);

input_training = zeros(sizee, num_training);
label_training = zeros(1, num_training);

input_testing = zeros(sizee, num_testing);
label_testing = zeros(1, num_testing);

%%%%% 99999 pre-processing %%%%%%%%%%%%%%%%%%%%
for i = 1:num_training
    img = readimage(training, i);
    img = imbinarize(img);
    img = 1 - img; 
    input_training(:, i) = img(:);
    label_training(i) = single(training.Labels(i));
end

for i = 1:num_testing
    img = readimage(testing, i);
    img = imbinarize(img);
    img = 1 - img; 
    input_testing(:, i) = img(:);
    label_testing(i) = single(testing.Labels(i));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_classes = 7;

label_training_onehot = full(ind2vec(label_training, num_classes));
label_testing_onehot = full(ind2vec(label_testing, num_classes));

hidden_layer = [128, 64];  
net = patternnet(hidden_layer);

net.layers{3}.transferFcn = 'softmax';

net.trainParam.lr = 1e-3; 

net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};

net.trainParam.epochs = 600;
net.trainParam.max_fail = 2000;
net.trainParam.goal = 0; 
net.trainParam.time = 300;
net.trainParam.min_grad = 1e-100;

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:num_training;
net.divideParam.valInd = [];
net.divideParam.testInd = [];

[net, tr] = train(net, input_training, label_training_onehot, 'useGPU', 'yes');
%[net, tr] = train(net, input_training, label_training_onehot);

test_ok = net(input_testing);

predicted_labels = vec2ind(test_ok);

correct_rate = sum(predicted_labels == label_testing) / num_testing;
disp(['Accuracy: ', num2str(correct_rate * 100), '%']);

save('mlp_test.mat', 'net');