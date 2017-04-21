function [Theta1, Theta2] = unrollParams(nn_params, ...
  input_layer_size, hidden_layer_size, num_labels)

  t_boundary = hidden_layer_size * (input_layer_size + 1);

  Theta1 = reshape(nn_params(1:t_boundary), ...
    hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((t_boundary+1):end), ...
    num_labels, (hidden_layer_size + 1));

end
