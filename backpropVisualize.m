function [J grad] = backpropVisualize(nn_params, input_layer_size, ...
  hidden_layer_size, num_labels, X, Y, lambda)

  [Theta1 Theta2] = unrollParams(nn_params, input_layer_size, ...
   hidden_layer_size, num_labels);

  display_array = calcVisualizatonMatrix(Theta1(:, 2:end));
  d_img = mat2gray(display_array, [-1 1]);
  d_img = imresize(d_img, 3);
  imwrite(d_img, strcat("./output/out", num2str(cputime), ".gif"), "gif")

  [J grad] = backprop(nn_params, input_layer_size, hidden_layer_size, num_labels, ...
   X, Y, lambda);
end
