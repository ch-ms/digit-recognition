% Settings
testGrads = false;

% Load dataset
disp("Loading data");

load('ex4data1.mat');
m = size(X, 1);

% Visualize
sel = randperm(size(X, 1));
sel = sel(1:100);
v_X = X(sel, :);
v_y = y(sel);

visualize(v_X);
anykey;

% Initialize
input_layer_size  = size(X, 2);
hidden_layer_size = 25;
num_labels = 10;

[train_X, test_X, cv_X, train_y, cv_y, test_y] = splitData([X y], unique(y));
train_Y = yMatrixForm(train_y, num_labels);
cv_Y = yMatrixForm(cv_y, num_labels);
test_Y = yMatrixForm(test_y, num_labels);
Y = yMatrixForm(y, num_labels);

fprintf("Train set size: %i\n", size(train_X)(1));
fprintf("Test set size: %i\n", size(test_X)(1));
fprintf("CV set size: %i\n\n", size(cv_X)(1));

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = rollParams(initial_Theta1, initial_Theta2);

% Test gradients
if (testGrads)
  disp("Check gradient");
  lambda = 0;
  testX = X(1, :);
  testY = Y(1, :);
  cfn = @(p) costFnByTheta(p, input_layer_size, hidden_layer_size, num_labels, ...
   testX, testY, lambda);

  testNumGrad = computeNumericalGradient(initial_nn_params, cfn);
  [testCost testRealGrad] = backprop(initial_nn_params, input_layer_size, ...
   hidden_layer_size, num_labels, testX, testY, lambda);

  disp([testNumGrad testRealGrad]);
  diff = norm(testNumGrad-testRealGrad)/norm(testNumGrad+testRealGrad);
  disp(diff);
  anykey();
end


% Learn
disp("Learning");
options = optimset('MaxIter', 200);
lambda = 1;

start_cost = costFnByTheta(initial_nn_params, input_layer_size, hidden_layer_size, ...
 num_labels, train_X, train_Y, lambda);

fprintf("Start cost: %i\n", start_cost);

bp = @(p) backprop(p, input_layer_size, hidden_layer_size, num_labels, ...
 train_X, train_Y, lambda);

[nn_params end_cost] = fmincg(bp, initial_nn_params, options);

[Theta1 Theta2] = unrollParams(nn_params, input_layer_size, hidden_layer_size, ...
 num_labels);

anykey;

% Visualize hidden layer

disp("Visualizing hidden layer");

disp("Theta 1")
visualize(Theta1(:, 2:end));
anykey;

disp("Theta 2");
visualize(Theta2(:, 2:end));

% for exmpl = 1:size(X)(1)
%   fprintf("Theta 1 for %i\n", v_y(exmpl));
%   th1 = initial_Theta1(:, 2:end);
%   visualize(th1 .* v_X(exmpl, :));
%   anykey;
% end

anykey;

% Predict
pred_train = predict(Theta1, Theta2, train_X);
pred_test = predict(Theta1, Theta2, test_X);
pred_cv = predict(Theta1, Theta2, cv_X);

fprintf("Training set accuracy: %f\n", mean(double(pred_train == train_y)) * 100);
fprintf("Test set accuracy: %f\n", mean(double(pred_test == test_y)) * 100);
fprintf("CV set accuracy: %f\n", mean(double(pred_cv == cv_y)) * 100);
