syms t I(t)

eq = 30*cos(t) == diff(I(t), t) +5*I(t);

cond= I(0) == 0;

sol = dsolve(eq, cond);
x = double(subs(sol, t, 0.5));

%Plot von Chat GPT erstellt
figure
fplot(sol, [0,10], 'LineWidth', 2) %
xlabel('t')
ylabel('I(t)')