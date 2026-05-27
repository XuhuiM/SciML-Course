function dydt = odeefun(t, y)
a = 1.5;
dydt = sin(3*pi*t) + a*y.*(1 - y);
end

