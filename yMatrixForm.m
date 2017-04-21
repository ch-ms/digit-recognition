function Y = yMatrixForm(y, num_labels)
  m = length(y);
  Y = zeros(m, num_labels);

  for i = 1:length(y)
    Y(i, y(i)) = 1;
  end

end
