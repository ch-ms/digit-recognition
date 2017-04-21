function [h, display_array] = visualize(X)

  % display
  colormap(gray);
  h = imagesc(calcVisualizatonMatrix(X), [-1 1]);
  axis image off
  drawnow;

end
