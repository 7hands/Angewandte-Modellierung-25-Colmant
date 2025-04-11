%Numeric Math:
a = 2;
f = @(t,x) -a*x;
y0 = 1;

[t_num, y_num] = ode45(f,[0,10], y0);



% Symbolic Math:

syms x(t) a positive

eq = diff(x, t) == -a*x(t);

cond = x(0) == 1;

sol = dsolve(eq,cond);

figure
plot(t_num, y_num)
hold on
fplot(subs(sol, a, 2), [0, 10])