function J = costFn(Y, h)
  m = size(Y)(1);
  J = 1/m * sum(sum(-Y.*log(h) - (1 - Y).*log(1-h)));
end
