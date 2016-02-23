clear;

model = Network();
model = add(model, fcLayer('fc1',784,256,0.01,0.001,0.005,0.9));
model = add(model, Sigmoid('sigmoid1'));
model = add(model, fcLayer('fc2',256,10,0.01,0.001,0.005,0.9));
model = add(model, Sigmoid('sigmoid2'));
model = add(model, softmaxLossLayer('loss'));

load('F:/MATLAB_NN/MLP/mnist.mat');
mean_value = mean(train_x, 2);
train_x = bsxfun(@minus, train_x, mean_value);
test_x = bsxfun(@minus, test_x, mean_value);

model = Fit_mlp(model, train_x, train_y, test_x, test_y,...
	100, 20, 1, 20, 20);
