clear;

load ode_pred;

% f_pred = f_pred/38;

% load ../dataset/KdV_training;

NX = 100;

noise_u = 0.01;

noise_f = 0.05;

% u_pred = u_pred*8;

u_mean = mean(u_pred);

u_std = std(u_pred);

u_std = sqrt(u_std.^2 + noise_u^2);

f_mean = mean(f_pred);

f_std = std(f_pred);

f_std = sqrt(f_std.^2 + noise_f^2);

a_mean = mean(a_pred);

a_std = std(a_pred);

s = a_pred.*u_pred.*(1. - u_pred);

s_mean = mean(s);

s_std = std(s);

% s_std = sqrt(s_std.^2);

% x = t_pred';

x = t_test';

u_low = u_mean - 2*u_std;

u_high = u_mean + 2*u_std;

f_low = f_mean - 2*f_std;

f_high = f_mean + 2*f_std;

s_low = s_mean - 2*s_std;

s_high = s_mean + 2*s_std;

X = [x, fliplr(x)];

% for idx = 1:NT

%     clf;

hold on;

Y = [u_low, fliplr(u_high)];

hc = fill(X, Y, 'c', 'edgecolor', 'none');

ht = plot(t_u_train, u_train, 'bo', 'linewidth', 2.0);

hr = plot(t_test, u_test, 'k-', 'linewidth', 2.0);

hm = plot(x, u_mean, 'r--', 'linewidth', 2.0);

box on;

%     set(gca, 'fontsize', 14, 'ylim', [-0.1, 10]);

xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$u$', 'interpreter', 'latex', 'fontsize', 14);

set(gcf, 'pos', [100 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.84 0.84], 'fontsize', 14);

% xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);
% 
% ylabel('$\eta_y$', 'interpreter', 'latex', 'fontsize', 14);

hl = legend([hr, ht, hm, hc], '$\mbox{Reference}$', '$\mbox{Training}$', '$\mbox{Mean}$', '$\mbox{2 Std}$', 'location', 'northwest');

set(hl, 'interpreter', 'latex', 'fontsize', 14);

legend boxoff;

box on;

% end

%%%------------------------------------------------------------------
figure;

hold on;

Y = [f_low, fliplr(f_high)];

hc = fill(X, Y, 'c', 'edgecolor', 'none');

ht = plot(t_f_train, f_train, 'bo', 'linewidth', 2.0);

hr = plot(t_test, f_test, 'k-', 'linewidth', 2.0);

hm = plot(x, f_mean, 'r--', 'linewidth', 2.0);

box on;

%     set(gca, 'fontsize', 14, 'ylim', [-0.1, 10]);

xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$f$', 'interpreter', 'latex', 'fontsize', 14);

set(gcf, 'pos', [100 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.84 0.84], 'fontsize', 14);

% xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);
% 
% ylabel('$f_y$', 'interpreter', 'latex', 'fontsize', 14);

% hl = legend([hr, ht, hm, hc], '$\mbox{Reference}$', '$\mbox{Training}$', '$\mbox{Mean}$', '$\mbox{2 Std}$', 'location', 'north');
% 
% set(hl, 'interpreter', 'latex', 'fontsize', 14);
% 
% legend boxoff;

box on;

% end

%%%%%%%%%%%%%%%%%%%%%%%%

figure;

hold on;

s_test = 1.5.*u_test.*(1 - u_test);

Y = [s_low, fliplr(s_high)];

hc = fill(X, Y, 'c', 'edgecolor', 'none');

% ht = plot(t_f_train, f_train, 'bo', 'linewidth', 2.0);

hr = plot(t_test, s_test, 'k-', 'linewidth', 2.0);

hm = plot(x, s_mean, 'r--', 'linewidth', 2.0);

box on;

%     set(gca, 'fontsize', 14, 'ylim', [-0.1, 10]);

% xlabel('$x$', 'interpreter', 'latex', 'fontsize', 14);
% 
% ylabel('$s_y$', 'interpreter', 'latex', 'fontsize', 14);

set(gcf, 'pos', [100 100 500 400]);

set(gca, 'pos', [0.14 0.14 0.84 0.84], 'fontsize', 14);

xlabel('$t$', 'interpreter', 'latex', 'fontsize', 14);

ylabel('$s$', 'interpreter', 'latex', 'fontsize', 14);

% hl = legend([hr, ht, hm, hc], '$\mbox{Reference}$', '$\mbox{Training}$', '$\mbox{Mean}$', '$\mbox{2 Std}$', 'location', 'north');
% 
% set(hl, 'interpreter', 'latex', 'fontsize', 14);
% 
% legend boxoff;

box on;