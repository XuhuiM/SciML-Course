load ode_pred; load ode_data;

xc = 0.5;

hold on;

h_ref = plot(t, y, 'k-', 'linewidth', 2.0);

h1 = plot(t_test_1, u_test_1, 'b--', 'linewidth', 2.0);

h2 = plot(t_test_2, u_test_2, 'r--', 'linewidth', 2.0);

h_t_1 = plot(t_u_train_1, u_train_1, 'bo', 'linewidth', 2.0);

h_t_2 = plot(t_u_train_2, u_train_2, 'bo', 'linewidth', 2.0);

hc = plot([xc xc], [0, 0.4], 'k--', 'linewidth', 2.0);

hl = legend([h_ref, h1, h2, h_t_1, hc], '$\mbox{reference}$', '$\mbox{PINN-1}$', '$\mbox{PINN-2}$', '$\mbox{Training data}$', '$\mbox{Coupling boundary}$', 'location', 'northwest');

set(hl, 'interpreter', 'latex', 'fontsize', 14);

legend boxoff;

xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$u$', 'interpreter', 'latex', 'fontsize', 14);

set(gcf, 'pos', [100, 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.82 0.82], 'fontsize', 14);

box on;

figure;

hold on;

h_ref = plot(t, f, 'k-', 'linewidth', 2.0);

h1 = plot(t_test_1, f_test_1, 'b--', 'linewidth', 2.0);

h2 = plot(t_test_2, f_test_2, 'r--', 'linewidth', 2.0);

h_t_1 = plot(t_f_train_1, f_train_1, 'bo', 'linewidth', 2.0);

h_t_2 = plot(t_f_train_2, f_train_2, 'bo', 'linewidth', 2.0);

hc = plot([xc xc], [-1.5, 0.], 'k--', 'linewidth', 2.0);

hl = legend([h_ref, h1, h2, h_t_1, hc], '$\mbox{reference}$', '$\mbox{PINN-1}$', '$\mbox{PINN-2}$', '$\mbox{Training data}$', '$\mbox{Coupling boundary}$', 'location', 'north');

set(hl, 'interpreter', 'latex', 'fontsize', 14);

legend boxoff;

xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$f$', 'interpreter', 'latex', 'fontsize', 14);

set(gcf, 'pos', [100, 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.82 0.82], 'fontsize', 14);

box on;