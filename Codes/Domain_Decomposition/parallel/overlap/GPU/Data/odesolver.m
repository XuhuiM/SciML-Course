clear;

tspan = [0, 1];

u0 = 0.;

sol = ode45(@odefun, tspan, u0);

t = linspace(0, 1, 101);

u = deval(sol, t);

f = sin(3*pi*t);

s = 1.5*u.*(1 - u);

figure;

plot(t, u, 'k-', 'linewidth', 2.0);

figure;

plot(t, f, 'k-', 'linewidth', 2.0);

figure;

plot(t, s, 'k-', 'linewidth', 2.0);

save('ode_data', 't', 'u', 'f', 's');