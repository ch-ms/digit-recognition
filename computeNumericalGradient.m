function grad = computeNumericalGradient(theta, J)

  epsilon = 1e-4;
  grad = zeros(size(theta));

  for i = 1:numel(theta)
    tp = theta;
    tm = theta;
    tp(i) = tp(i) + epsilon;
    tm(i) = tm(i) - epsilon;
    grad(i) = (J(tp) - J(tm))/(2*epsilon);
  end

end
