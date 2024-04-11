data = load('viv_raw_data');

t0 = 840;

idx= 2;

t = data(1:idx:end, 1) - 840;

eta_x = data(1:idx:end, 2);

eta_y = data(1:idx:end, 3);

f_x = data(1:idx:end, 10);

f_y = data(1:idx:end, 11);

save('VIV_Training', 't', 'eta_y', 'f_y');

figure;

% subplot(121);

plot(t, eta_y, 'r-', 'linewidth', 2.0);

% subplot(122);

figure;

plot(t, f_y, 'k-', 'linewidth', 2.0);