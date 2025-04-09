clear;
clc;
close all;

training = imageDatastore('p_dataset_26\dataset\train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) imbinarize(imresize(imcomplement(imread(x)), [128, 128]))); 

testing = imageDatastore('p_dataset_26\dataset\test', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) imbinarize(imresize(imcomplement(imread(x)), [128, 128])));

num_classes = 7;
imageSize = [128 128 1];  

layers = [
    imageInputLayer([128 128 1], 'Normalization', 'zerocenter', 'Name', 'imageinput') 
    
    convolution2dLayer(3, 8, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_1')  
    batchNormalizationLayer('Name', 'batchnorm_1') 
    reluLayer('Name', 'relu_1')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_1')  
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_2')  
    batchNormalizationLayer('Name', 'batchnorm_2')  
    reluLayer('Name', 'relu_2')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_2')  
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'batchnorm_3')  
    reluLayer('Name', 'relu_3')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_3')  
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_4')  
    batchNormalizationLayer('Name', 'batchnorm_4')  
    reluLayer('Name', 'relu_4')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_4')  
    
    convolution2dLayer(3, 128, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_5')  
    batchNormalizationLayer('Name', 'batchnorm_5')  
    reluLayer('Name', 'relu_5')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_5')  
    
    convolution2dLayer(3, 256, 'Padding', 'same', 'Stride', [1 1], 'Name', 'conv_6')  
    batchNormalizationLayer('Name', 'batchnorm_6')  
    reluLayer('Name', 'relu_6')  
    maxPooling2dLayer(2, 'Stride', [2 2], 'Padding', [0 0 0 0], 'Name', 'maxpool_6')  
    
    fullyConnectedLayer(512, 'Name', 'fc_1')  
    reluLayer('Name', 'relu_7')  
    
    fullyConnectedLayer(num_classes, 'Name', 'fc_2')  
    softmaxLayer('Name', 'softmax') 
    classificationLayer('Name', 'classoutput')  
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 15, ... 
    'Shuffle','every-epoch', ...
    'MiniBatchSize', 64, ...  
    'Shuffle', 'every-epoch', ... 
    'ValidationData', testing, ... 
    'ValidationFrequency', 20, ...  
    'Verbose', false, ... 
    'Plots', 'training-progress', ...  
    'ExecutionEnvironment', 'gpu');  


net = trainNetwork(training, layers, options);

YPred = classify(net, testing);
YTest = testing.Labels;

accuracy = sum(YPred == YTest)/numel(YTest);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

save('cnn_test.mat', 'net');
