function [J grad] = backprop(nn_params, input_layer_size, ...
  hidden_layer_size, num_labels, X, Y, lambda)

  m = size(Y)(1);

  [Theta1, Theta2] = unrollParams(nn_params, input_layer_size, ...
    hidden_layer_size, num_labels);

  ones_X = [ones(size(X)(1), 1) X];
  a1 = ones_X;
  [z2, z3, a2, a3] = forwardprop(Theta1, Theta2, X);

  % section: COSTFN
  J = costFn(Y, a3);

  % section: GRADIENT
  delta3 = (a3 - Y)';
  DELTA2 = delta3 * a2;

  delta2 = (Theta2' * delta3) .* sigmoidGradient(z2');
  DELTA1 = delta2(2:end, :) * a1;

  Theta1_grad = 1/m .* DELTA1;
  Theta2_grad = 1/m .* DELTA2;

  % grad for all examples
  grad = rollParams(Theta1_grad, Theta2_grad);

end
