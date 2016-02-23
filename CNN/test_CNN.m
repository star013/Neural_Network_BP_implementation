%  =========================================================
%  A typical architecture of CNN is organized as follows:
%
%   - add convlayer to model
%   - add relu to model
%   - add poolLayer to model
%
%  and then repeat the above mini-structure as many times as you like.
%  
%  Before the final classification operation
%
%   - add fcLayer to model
%
%  and then add softmaxLossLayer
%
%
%  ATTENTION!
%    
%    fcLayer should implement matrix reshape operation to align the bottom input
%    as a 2D matrix with dimension (H * W * C) x N
%  ==========================================================

clear;
model = Network();
% weight initiation standard variation should be larger in larger dataset
% 0.1 is far better than 0.01 for example when trainset has 1000 images
model = add(model, convLayer('conv1',3,1,8,1,0.1,0.001,0.005,0.9));
model = add(model, Relu('relu1'));
model = add(model, poolLayer('pool1',2,0));
model = add(model, convLayer('conv2',3,8,4,1,0.1,0.001,0.005,0.9));
model = add(model, Relu('relu2'));
model = add(model, poolLayer('pool2',2,0));
model = add(model, fcLayer('fc3',196,10,0.01,0.001,0.005,0.9));
model = add(model, softmaxLossLayer('softmax'));

load('F:/MATLAB_NN/MLP/mnist.mat');
mean_value = mean(train_x, 2);
train_x = bsxfun(@minus, train_x, mean_value);
test_x = bsxfun(@minus, test_x, mean_value);

%test
% temp_x=zeros(784,100);
% for k=1:100
%     temp_x(:,k) = train_x(:,1);
% end
% temp_y = train_y(1)*ones(1,100);

% temp_x = train_x(:,1:500);
% temp_y = train_y(1:500);
% 
% train_x = temp_x;
% train_y = temp_y;
% test_x = test_x(:,1:500);
% test_y = test_y(1:500);

train_x = permute(reshape(train_x, 28, 28, 1, []), [2, 1, 3, 4]);
test_x = permute(reshape(test_x, 28, 28, 1, []), [2, 1, 3, 4]);
%%
model = Fit_cnn(model, train_x, train_y, test_x, test_y,...
    100, 20, 1, 10, 100);