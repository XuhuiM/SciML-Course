function dudx = ode(x, u, fx, f)

f = interp1(fx, f, x);

dudx = f + 0.2*u.^3;

end