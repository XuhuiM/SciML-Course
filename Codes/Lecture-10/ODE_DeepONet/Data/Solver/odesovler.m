%%%%%%--------ODE solver---------------------------------------------
%%%%%%--------du/dx = f-----------------------------------------------
%%%%%%--------f ~ GP(0, K(x, x'))---------------------------------------
%%%%%%--------Author: Xuhui Meng-----------------------------------
%%%%%%--------Email: xuhmeng@gmail.com--------------------------
clear;

rng('default');

N = 50;

x = linspace(-1, 1, N);

u0 = 0;

fx = linspace(-1, 1, N);

%%%-------------GP for f---------------------------------------------------------
l = 0.2;

len = length(fx);

cov = zeros(len, len);

for i = 1:len
    for j  = 1:len
        cov(i, j) = exp(-0.5*(fx(i) - fx(j)).*(fx(i) - fx(j))/l/l);
    end
end

mu = zeros(1, len);

num = 10000;

for i = 1:num
    
    f = mvnrnd(mu, cov, 1);

    [x, u] = ode45(@(x, u) ode(x, u, fx, f), x, u0);
    
    F(i, :) = f;
    
    U(i, :) = u';
    
end

filename = 'ODE_Train_Data';

x_train = x;

save(filename, 'F', 'U', 'x_train');