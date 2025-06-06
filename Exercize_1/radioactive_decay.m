%Numeric Math:
a = 2; %für die numerische Berechnung muss alpha fest sein
f = @(t,x) -a*x; % die funktion zur numerischen berechnung
y0 = 1; % die bedingung

[t_num, y_num] = ode45(f,[0,10], y0); %berechnung



% Symbolic Math:

syms x(t) a positive %Symbole

eq = diff(x, t) == -a*x(t); % Die Differential Gleichung

cond = x(0) == 1; %Die Bedinungen

sol = dsolve(eq,cond); % Berechnung

%Plot von Chat GPT erstellt
figure
plot(t_num, y_num(:,1), 'b', 'LineWidth', 2)
hold on
fplot(subs(sol, a, 2), [0,10], 'r--', 'LineWidth', 2) %
legend('Numerisch (ode45)', 'Symbolisch')
xlabel('t')
ylabel('x(t)')
title('Vergleich: symbolisch vs. numerisch')
grid on
