function J = costFnByTheta(nn_params, input_layer_size, hidden_layer_size, num_labels, ...
 X, Y, lambda)

 [Theta1, Theta2] = unrollParams(nn_params, input_layer_size, ...
  hidden_layer_size, num_labels);

 [z2, z3, a2, h] = forwardprop(Theta1, Theta2, X);

 J = costFn(Y, h, Theta1, Theta2, lambda);

end
