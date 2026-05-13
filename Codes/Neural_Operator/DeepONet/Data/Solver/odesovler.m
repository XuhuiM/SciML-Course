%%%%%%--------ODE solver---------------------------------------------
%%%%%%--------f ~ GP(0, K(x, x'))---------------------------------------
%%%%%%--------Author: Xuhui Meng-----------------------------------
%%%%%%--------Email: xuhui_meng@hust.edu.cn--------------------------
clear;

rng('default');

N = 500;

x = linspace(-1, 1, N);

u0 = 0;

fx = linspace(-1, 1, N);

%%%-------------GP for f---------------------------------------------------------
l = 0.1;

len = length(fx);

cov = zeros(len, len);

for i = 1:len
    for j  = 1:len
        cov(i, j) = exp(-0.5*(fx(i) - fx(j)).*(fx(i) - fx(j))/l/l);
    end
end

mu = zeros(1, len);

num = 1;

for i = 1:num
    
    f = mvnrnd(mu, cov, 1);

    [x, u] = ode45(@(x, u) ode(x, u, fx, f), x, u0);
    
    F(i, :) = f;
    
    U(i, :) = u';
    
end

filename = 'ODE_Train_Data';

x_train = x;

save(filename, 'F', 'U', 'x_train');