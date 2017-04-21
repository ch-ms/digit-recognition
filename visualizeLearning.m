pkg load image
load('ex4data1.mat');
m = size(X, 1);

% Initialize
input_layer_size  = size(X, 2);
hidden_layer_size = 25;
num_labels = 10;

[train_X, test_X, cv_X, train_y, cv_y, test_y] = splitData([X y], unique(y));
train_Y = yMatrixForm(train_y, num_labels);
cv_Y = yMatrixForm(cv_y, num_labels);
test_Y = yMatrixForm(test_y, num_labels);
Y = yMatrixForm(y, num_labels);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = rollParams(initial_Theta1, initial_Theta2);

disp("Learning");
options = optimset('MaxIter', 200);
lambda = 1;

start_cost = costFnByTheta(initial_nn_params, input_layer_size, hidden_layer_size, ...
 num_labels, train_X, train_Y, lambda);

fprintf("Start cost: %i\n", start_cost);

bp = @(p) backpropVisualize(p, input_layer_size, hidden_layer_size, num_labels, ...
 train_X, train_Y, lambda);

[nn_params end_cost] = fmincg(bp, initial_nn_params, options);

[Theta1 Theta2] = unrollParams(nn_params, input_layer_size, hidden_layer_size, ...
 num_labels);

anykey;
