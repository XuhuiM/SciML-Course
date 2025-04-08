tspan = [0, 1];

y0 = 0.5;

sol = ode45(@odefun, tspan, y0);

t = linspace(0, 1, 101);

y = deval(sol, t);

f = sin(3*pi*t);

s = 1.5*y.*(1 - y);

figure;

plot(t, y, 'k-', 'linewidth', 2.0);

figure;

plot(t, f, 'k-', 'linewidth', 2.0);

figure;

plot(t, s, 'k-', 'linewidth', 2.0);

save('ode_data', 't', 'y', 'f', 's');