%-----discrete points for solution------------------------------------------------
num_x = 101; x = linspace(-1, 1, num_x);
num_t = 101; t = linspace(0, 1, num_t);
[x_2d, t_2d] = meshgrid(x, t);

m = 0;

sol = pdepe(m, @cdeeq, @cdeic, @cdebc, x, t);

u = sol(:, :, 1);

figure;

imagesc(x, t, u); axis equal xy; colormap(jet);

xlabel('x'); ylabel('t');

colorbar;

figure;

plot(x, u', 'linewidth', 2.0);

xlabel('x'); ylabel('u');

figure;

% kr = exp((x - 0.1).^2).*cos(3*x).^2;

lambda = 0.2 + exp(x).*cos(2*x).^2;

plot(x, lambda, 'linewidth', 2.0);

xlabel('x'); ylabel('$\lambda$', 'interpreter', 'latex');

figure;

l = 0.4;

f = exp(-0.5*(x-0.25).^2/l/l).*sin(3*x).^2;

% f = 0.5*(exp(-0.5*(x - 0.25).^2/l^2) + exp(-0.5*(x + 0.25).^2/l^2));

plot(x, f, 'linewidth', 2.0);

xlabel('x'); ylabel('f');

save('./Project_III_Data', 'x', 't', 'lambda', 'f', 'x_2d', 't_2d', 'u');


%%%--PDE-----------------------------------------------------------
function [c, f, s] = cdeeq(x, t, u, dudx)
c = 1;
f = 0.01*dudx;
%     kr = -exp((x - 0.1).^2).*cos(3*x).^2;
lambda = 0.2 + exp(x).*cos(2*x).^2;
%     kr = 0.0;
l = 0.4;
s = -lambda.*u.^3 + (exp(-0.5*(x - 0.25).^2/l^2)).*sin(3*x).^2;
end

%%%---Boundary conditions: u_l = u_r = 1---------------------------
function [pl, ql, pr, qr] = cdebc(xl, ul, xr, ur, t)
pl = ul - 0.5;
ql = 0;
pr = ur - 0.5;
qr = 0;
end

%%%--Initiational conditions: u(x, 0) = cos(pi*x)^2-------------------------
function u0 = cdeic(x)
% u0 = cos(pi*x).*cos(pi*x);
u0 = 0.5*cos(pi*x).^2;
end