function dudx = ode(x, u, fx, f)

f = interp1(fx, f, x);

dudx = f;

end