function J = costFn(Y, h, Theta1, Theta2, lambda)
  m = size(Y)(1);
  J = 1/m * sum(sum(-Y.*log(h) - (1 - Y).*log(1-h)));

  % Calc regularization
  reg_theta1 = Theta1(:, 2:size(Theta1)(2)).^2;
  reg_theta2 = Theta2(:, 2:size(Theta2)(2)).^2;

  J = J + (lambda/(2*m) * ( sum(sum(reg_theta1)) + sum(sum(reg_theta2)) ));
end
