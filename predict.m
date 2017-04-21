function p = predict(Theta1, Theta2, X)

  [z2, z3, a2, h] = forwardprop(Theta1, Theta2, X);
  [dummy, p] = max(h, [], 2);

end
