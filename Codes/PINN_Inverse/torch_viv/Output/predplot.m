load('flow_match_pred'); 

hold on;

histogram2(samples(:, 1),samples(:, 2), 'DisplayStyle','tile','ShowEmptyBins','on');

axis equal; colormap(jet);
% plot(pred_1.x_train, pred_1.y_train, 'm*', 'linewidth', 2.0);
% 
% plot(pred_1.x_test, pred_1.y_pred, 'y:', 'linewidth', 2.0);

% hl = legend([h1, h2, hc, h3], '$\mbox{Exact}$', '$\mbox{Mean}$', '$\mbox{2 ~Std}$', '\mbox{Training}', 'location', 'northeast');

% set(hl, 'fontsize', 16, 'interpreter', 'latex');

xlabel('$x$', 'fontsize', 16, 'interpreter', 'latex'); ylabel('$y$', 'fontsize', 16, 'interpreter', 'latex');

% legend boxoff;

box on;

% err = (pred_0.y_pred - pred_1.y_pred);

figure;

[x_dim, y_dim, z_dim] = size(v_pred); 

v_pred_0 = reshape(v_pred(:, :, 1), x_dim, y_dim);

v_ref_0 = reshape(u_ref(:, :, 1), x_dim, y_dim);

imagesc(v_pred_0); axis xy off; colormap(jet);

figure;

imagesc(v_ref_0); axis xy off; colormap(jet);

err_0 = norm(v_pred_0 - v_ref_0)/norm(v_ref_0)