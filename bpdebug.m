load('ex4data1.mat');

input_layer_size  = size(X, 2);
hidden_layer_size = 25;
num_labels = 10;

Y = yMatrixForm(y, num_labels);

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

nn_params = rollParams(Theta1, Theta2);

a1 = [ones(size(X)(1), 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
z2 = [ones(size(z2)(1), 1) z2];
a2 = [ones(size(a2)(1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

disp(size(X));
disp(size(a2));
disp(size(a3));
disp("Y=")
disp(size(Y))
disp("Theta2=")
disp(size(Theta2))

backprop(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, 0)
