pred_0 = load('y_pred_0'); 

pred_1 = load('y_pred_1');

hold on;

plot(pred_0.x_test, pred_0.y_ref, 'k-', 'linewidth', 2.0);

plot(pred_0.x_train, pred_0.y_train, 'bo', 'linewidth', 2.0);

plot(pred_0.x_test, pred_0.y_pred, 'r--', 'linewidth', 2.0);

plot(pred_1.x_train, pred_1.y_train, 'm*', 'linewidth', 2.0);

plot(pred_1.x_test, pred_1.y_pred, 'y:', 'linewidth', 2.0);

% hl = legend([h1, h2, hc, h3], '$\mbox{Exact}$', '$\mbox{Mean}$', '$\mbox{2 ~Std}$', '\mbox{Training}', 'location', 'northeast');

% set(hl, 'fontsize', 16, 'interpreter', 'latex');

xlabel('$x$', 'fontsize', 16, 'interpreter', 'latex'); ylabel('$y$', 'fontsize', 16, 'interpreter', 'latex');

legend boxoff;

box on;

err = (pred_0.y_pred - pred_1.y_pred);