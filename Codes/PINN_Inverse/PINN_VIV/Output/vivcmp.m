load ../Data/VIV_Training;

load VIV_Pred;

hold on;

te = 160;

h1 = plot(t(1:te), eta_y(1:te), 'k-', 'linewidth', 2.0);

h2 = plot(t_u_train, u_train, 'bo', 'linewidth', 2.0);

h3 = plot(t_test, u_test, 'r--', 'linewidth', 2.0);

set(gcf, 'pos', [100 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.84 0.84], 'fontsize', 14);

xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$\eta_y$', 'interpreter', 'latex', 'fontsize', 14);

hl = legend([h1, h2, h3], '$\mbox{Numerical}$', '$\mbox{Training}$', '$\mbox{PINN}$', 'location', 'northwest');

set(hl, 'interpreter', 'latex', 'fontsize', 14);

legend boxoff;

box on;

figure;

hold on;

h1 = plot(t(1:te), f_y(1:te), 'k-', 'linewidth', 2.0);

h2 = plot(t_f_train, f_train, 'bo', 'linewidth', 2.0);

h3 = plot(t_test, f_test, 'r--', 'linewidth', 2.0);

set(gcf, 'pos', [100 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.84 0.84], 'fontsize', 14);

xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$f_y$', 'interpreter', 'latex', 'fontsize', 14);

hl = legend([h1, h2, h3], '$\mbox{Numerical}$', '$\mbox{Training}$', '$\mbox{PINN}$', 'location', 'north');

set(hl, 'interpreter', 'latex', 'fontsize', 14);

legend boxoff;

box on;