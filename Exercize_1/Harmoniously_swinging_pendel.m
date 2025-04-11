%Symbolic Math:
syms x(t) omega positive % Symbole

eq = diff(x, t, 2) + omega^2 * x == 0; % Equation

Dx = diff(x, t); % first differential

cond = [x(0) == 0, Dx(0) == 1]; % Conditions for the Equation

sol = dsolve(eq, cond);% 


% numeric Math:
% Parameter
omega_val = 1;  % Setze z.B. omega = 2

% System 1. Ordnung: x1 = x, x2 = dx/dt
f = @(t, y) [y(2); -omega_val^2 * y(1)];

% Anfangswerte: x(0) = 0, dx/dt(0) = 1
y0 = [0; 1];

% Zeitintervall
tspan = [0, 5];

% Numerische Lösung
[t_num, y_num] = ode45(f, tspan, y0);

% Plot
figure
plot(t_num, y_num(:,1), 'b', 'LineWidth', 2)
hold on
fplot(subs(sol, omega, omega_val), tspan, 'r--', 'LineWidth', 2)
legend('Numerisch (ode45)', 'Symbolisch')
xlabel('t')
ylabel('x(t)')
title('Vergleich: symbolisch vs. numerisch')
grid on