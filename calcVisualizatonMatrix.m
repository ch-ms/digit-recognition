function display_array = calcVisualizatonMatrix(X)

    [m n] = size(X);

    example_width = round(sqrt(n));
    example_height = n / example_width;

    display_rows = floor(sqrt(m));
    display_cols = ceil(m / display_rows);

    pad = 1;

    display_array = - ones(
      pad + display_rows * (example_height + pad),
      pad + display_cols * (example_width + pad)
    );

    % copy examples
    curr_ex = 1;
    for j = 1:display_rows
      for i = 1:display_cols

        if(curr_ex <= m)

          ex = reshape(X(curr_ex, :), example_height, example_width);
          ex = flipud(ex);

          max_val = max(abs(X(curr_ex, :)));
          ex = ex / max_val;

          display_array(
            pad + (j - 1) * (example_height + pad) + (1:example_height), ...
            pad + (i - 1) * (example_width + pad) + (1:example_width)
          ) = ex;

          curr_ex = curr_ex + 1;
        end

      end
    end

end
