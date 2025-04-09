clc; clear; close all;

%% Req 1 - Display original image and experimenting with contrast
% Load the .BMP file and display the original image
PSY = imread('charact2.bmp');  % Load the Image in :)
figure(1);                     
imshow(PSY);                   % Display the Original Image
title("Original Image");

% Convert Image to Grayscale from RGB
gray_PSY = rgb2gray(PSY);

% Method 1: Adjust image intensity value with default contrast enhancement
adjusted_gray_PSY = imadjust(gray_PSY);

% Method 2: Provide and adjust input/output intensity range from histogram
%imhist(gray_PSY);            % Display histogram of grayscale image
intensity_gray_PSY = imadjust(gray_PSY, [0.19 0.51], [0 1]); % Change the range for intensity from histogram

% Method 3: Use Histogram Equalization to improve contrast in low-contrast images
equalized_gray_PSY = histeq(gray_PSY);

% Method 4: Use Adaptive Histogram Equalization to improve local contrast
adaptive_equalized_gray_PSY = adapthisteq(gray_PSY);

% Plot the different contrast enhancement methods
figure(2);
sgtitle("Requirement 1");
subplot(2,2,1), imshow(adjusted_gray_PSY), title('Base Adjusted Grayscale');
subplot(2,2,2), imshow(intensity_gray_PSY), title('Adjusted Input/Output Range Grayscale Image');              % Display adjusted grayscale image
subplot(2,2,3), imshow(equalized_gray_PSY), title('Contrast Enhanced Image using Histogram Equalization');
subplot(2,2,4), imshow(adaptive_equalized_gray_PSY), title('Contrast Enhanced Image using Adaptive Histogram Equalization');

figure(3);
sgtitle("Requirement 1 Histogram");
subplot(2,2,1), imhist(adjusted_gray_PSY), title('Base Adjusted Grayscale');
subplot(2,2,2), imhist(intensity_gray_PSY), title('Adjusted Input/Output Range Grayscale Image');              % Display adjusted grayscale image
subplot(2,2,3), imhist(equalized_gray_PSY), title('Contrast Enhanced Image using Histogram Equalization');
subplot(2,2,4), imhist(adaptive_equalized_gray_PSY), title('Contrast Enhanced Image using Adaptive Histogram Equalization');

gray_PSY = adjusted_gray_PSY;

%% Req 2 - Implement and apply a 5x5 averaging filter
% Implementing a 5x5 Filter
a = 5; %change this number
ffffilter = ones(a,a)./a./a;
filter_PSY = imfilter(gray_PSY, ffffilter); %requirement 2 5*5 averaging filter

figure(4);
imshow(filter_PSY);
title("requirement 2");
filters = [3, 5, 7, 9] ; % change this number to experiment with different filter sizes

% Create loop to test different filter sizes
figure(5);
for k = 1:length(filters)
    a = filters(k);                 % Define filter size
    ffffilter = ones(a,a)./(a*a);    % Create filter
    filter_PSY = imfilter(gray_PSY, ffffilter); % Filter on Image
    
    % Plot the filtered images
    subplot(2, 2, k), sgtitle('Averaging Filters with Different Sizes');
    imshow(filter_PSY);
    title(sprintf('%dx%d Averaging Filter', a, a));
end

%% Req 3 - Implementing and Applying a high-pass filter
fly = fft2(double(gray_PSY)); % Applying Fourier Fat Transform (2D)
fftt = fftshift(fly);

[l, w] = size(gray_PSY); % Image Dimensions
[X, Y] = meshgrid(1:w, 1:l);
cX = ceil(w/2); % Center of X
cY = ceil(l/2); % Center of Y

% Create loop to test different cut-off frequencies
highps_range = [5, 10, 20, 25]; % Different Cut-off frequencies
number_filters = length(highps_range);
highpass_PSY = cell(1,number_filters);

figure(6);
for k = 1:number_filters
    highps = highps_range(k); % Cut-off Frequency
    highpass_filter = sqrt((X - cX).^2 + (Y - cY).^2) > highps;   % Create Filter
    highpass_fftt = fftt .* highpass_filter; % Implementing high-pass
    highpass_PSY{k} = real(ifft2(ifftshift(highpass_fftt))); % Conducting Inverse Fourier Trans. to get to spatial domain

    % Plot the comparisons
    subplot(2, 2, k), sgtitle('Req 3: Comparison of different cut-off frequency high-pass filtered images');
    imshow(highpass_PSY{k}, []);
    title(['High-Pass Filter, Cutoff = ', num2str(highps)]);
end

%% Req 4 - Create a sub-image of HD44780A00
figure(7);
sub_PSY1 = imcrop(gray_PSY, [1, 1, w, (l/2)-1]);
sub_PSY2 = imcrop(gray_PSY, [1, (l/2), w, (l/2)-1]);
imshow(PSY);  % Scale
subplot(2,1,1);
imshow(sub_PSY1);
title("Requirement 4 (up)");
subplot(2,1,2)
imshow(sub_PSY2);
title("Requirement 4 (down)");

%% Req 5 - Convert sub-image to binary
guass_PSY = imgaussfilt(gray_PSY, 20);  
high_freq_PSY = gray_PSY - guass_PSY;

sharpness_factor = 2;
sharpened_PSY = gray_PSY + sharpness_factor * high_freq_PSY;

contrast_PSY = histeq(sharpened_PSY);

bar = 0.839;
bi_PSY = imbinarize(contrast_PSY, bar);
bi_PSY = imcrop(bi_PSY, [1, (l/2), w, (l/2)-1]);
figure(8);
imshow(bi_PSY);
title("Requirement 5: Sub-image Binary")

%% Req 6 - Determine Outline of Characters
open_PSY = bwareaopen(bi_PSY, 185);

square1 = ones(5,5);
erode_PSY = imerode(open_PSY, square1);

square2 = ones(7,5);
erode_PSY = imdilate(erode_PSY,square2);

ud = ones(7,1);
erode_PSY = imerode(erode_PSY, ud);

ud = ones(13,1);
erode_PSY = imdilate(erode_PSY, ud);

edge_PSY = edge(erode_PSY, "canny");
figure(9);
imshow(edge_PSY);
title("Requirement 6")

%% Req 7 - Segment the image to separate characters
open_PSY = bwareaopen(edge_PSY, 185);
fill_PSY = imfill(open_PSY, 'holes');
%imshow(fill_PSY);

figure(10);
[labe, nummm] = bwlabel(fill_PSY);
numimages = cell(1, nummm);
numprops = regionprops(labe, 'BoundingBox');

for k = 1:nummm
    thisBoundingBox = numprops(k).BoundingBox;
    numimages{k} = imcrop(erode_PSY, thisBoundingBox);
    subplot(ceil(nummm/5), 5, k)
    imshow(numimages{k});
    title(["character ", num2str(k)])
end

%% Req 9 - pre-processing 128*128
figure(11);  
test_img = cell(1, nummm);

for ijk = 1:nummm
    [height, width] = size(numimages{ijk});  
    if height > 128
        start_row = floor((height - 128) / 2) + 1;
        numimages{ijk} = numimages{ijk}(start_row:start_row+127, :);
    end

    if width > 128
        start_col = floor((width - 128) / 2) + 1;
        numimages{ijk} = numimages{ijk}(:, start_col:start_col+127);
    end

    [height, width] = size(numimages{ijk});

    pad_height = max(0, 128 - height);
    pad_width = max(0, 128 - width);
    test_img{ijk} = padarray(numimages{ijk}, [floor(pad_height / 2), floor(pad_width / 2)], 0, 'both');

    if mod(pad_height, 2) ~= 0
        test_img{ijk} = padarray(test_img{ijk}, [1, 0], 0, 'post');  
    end

    if mod(pad_width, 2) ~= 0
        test_img{ijk} = padarray(test_img{ijk}, [0, 1], 0, 'post'); 
    end

    abc = ones(3,3);
    test_img{ijk} = imdilate(test_img{ijk}, abc);

    h = fspecial('gaussian', [5, 5], 2);
    test_img{ijk} = imfilter(double(test_img{ijk}), h);  
    test_img{ijk} = imbinarize(test_img{ijk}, 0.5);

    subplot(ceil(nummm/5), 5, ijk)
    imshow(test_img{ijk});
    title("128*128 pre-pro");
end

chaaa = {'H', 'D', '4', '4', '7', '8', '0', 'A', '0', '0'};
chaaab = string(chaaa);

%% MLP Classification Method
load('mlp.mat', 'net');
label_map = {'0', '4', '7', '8', 'A', 'D', 'H', 'M'} ;
sizee = 128 * 128; 
input_img_MLP = zeros(sizee, nummm);

for i = 1:nummm
    input_img_MLP(:, i) = test_img{i}(:);
end

char_pred_MLP = net(input_img_MLP);
predict_MLP = vec2ind(char_pred_MLP);

figure(10);
set(gcf, 'Position', [100, 100, 800, 600]);
for i = 1:nummm
    predict_num_MLP = predict_MLP(i);
    predict_letter_MLP = label_map{predict_num_MLP};
    acc_MLP = max(char_pred_MLP(:, i));
    accs_MLP = num2str(acc_MLP * 100, '%.2f');

    subplot(ceil(nummm/2), 2, i);
    imshow(test_img{i});
    if predict_letter_MLP == chaaab(i)
        title(['MLP: This is ', predict_letter_MLP, ' (', accs_MLP, ' %)' , '  Right! √'], 'Color', '[0 0.5 0]'); 
    else 
        title(['MLP: This is ', predict_letter_MLP, ' (', accs_MLP, ' %)' , '  Wrong! ×' ...
            ''], 'Color', 'red'); 
    end

end

%% CNN Classification Method
load('cnn.mat', 'net');
input_img_CNN = zeros(128, 128, 1, nummm);

for i = 1:nummm
    input_img_CNN(:, :, 1, i) = test_img{i};
end

char_pred_CNN = predict(net, input_img_CNN);
[~, predict_CNN] = max(char_pred_CNN, [], 2);

figure(11);
set(gcf, 'Position', [950, 100, 800, 600]);
for i = 1:nummm
    predict_num_CNN = predict_CNN(i);
    predict_letter_CNN = label_map{predict_num_CNN};
    acc_CNN = max(char_pred_CNN(i, :));
    accs_CNN = num2str(acc_CNN * 100, '%.2f');

    subplot(ceil(nummm/2), 2, i);
    imshow(test_img{i});
    if predict_letter_CNN == chaaab(i)
        title(['CNN: This is ', predict_letter_CNN, ' (', accs_CNN, ' %)' , '  Right! √'], 'Color', '[0 0.5 0]'); 
    else
        title(['CNN: This is ', predict_letter_CNN, ' (', accs_CNN, ' %)' , '  Wrong! ×'], 'Color', 'red'); 
    end
end

%% Test CNN and MLP by Rotating the Character
rot = test_img{9}; 
angle = 45;

rott = padarray(rot, [64, 64], 0, 'both');
rottt = imrotate(rott, angle, 'bilinear', 'crop');
rotttt = imcrop(rottt, [64 64 127 127]);

predict_num_CNN = classify(net, rotttt); 

pred_scores = predict(net, rotttt);
acc_CNN = max(pred_scores); 
accs_CNN = num2str(acc_CNN * 100, '%.2f');

figure
subplot(1, 2, 1);
imshow(rotttt);
title(['CNN: This is ', char(predict_num_CNN), ' (', accs_CNN, ' %)']);

input_data = rotttt(:); 
load('mlp.mat', 'net');

char_pred_MLP = net(input_data); 
predict_num_MLP = vec2ind(char_pred_MLP); 
predict_letter_MLP = label_map{predict_num_MLP}; 

acc_MLP = max(char_pred_MLP); 
accs_MLP = num2str(acc_MLP * 100, '%.2f');

subplot(1, 2, 2);
imshow(rotttt);
title(['MLP: This is ', predict_letter_MLP, ' (', accs_MLP, ' %)']);
