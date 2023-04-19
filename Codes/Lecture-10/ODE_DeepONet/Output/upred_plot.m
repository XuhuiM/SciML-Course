load ODE_Preds;

hold on;

idx = 20;

hold on;

plot(x_test, u_test(1:idx, :), 'k-', 'linewidth', 2.0);

plot(x_test, u_pred(1:idx, :), 'r--', 'linewidth', 2.0);

xlabel('$x$', 'fontsize', 14, 'interpreter', 'latex');

ylabel('$u$', 'fontsize', 14, 'interpreter', 'latex');

set(gcf, 'pos', [300 300 500 400]);

set(gca, 'pos', [0.14 0.14 0.82 0.82], 'fontsize', 14);

box on;
