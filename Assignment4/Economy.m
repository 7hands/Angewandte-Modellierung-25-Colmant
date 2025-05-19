% Parameter
C  = 1.4;
K  = 4;
G0 = 4;
f = @(t,x) [ x(1) - K*x(2);
             x(1) - C*x(2) - G0 ];

% Gitter
[Ig,Sg] = meshgrid(linspace(-50,50,30), linspace(-20,20,30));
dI = Ig - K*Sg;
dS = Ig - C*Sg - G0;

% Große Figur
figure('Units','pixels','Position',[100 100 1000 800]);
hold on; axis equal

% Vektorfeld
quiver(Ig, Sg, dI, dS, 'AutoScale','on','AutoScaleFactor',1.5, 'Color',[0.2 0.2 0.2]);

% Trajektorien
ics = [0 0; 5 2; 10 5; -10 10; 15 -5];
tspan = [0 15];
for i=1:size(ics,1)
    [t,X] = ode45(f, tspan, ics(i,:));
    plot(X(:,1), X(:,2), 'LineWidth',1.8);
end

% Achsen-Limits zum Zoomen
xlim([-50 50]);
ylim([-20 20]);

% Beschriftungen und Titel mit TeX-Interpreter
xlabel('I','FontSize',14);
ylabel('S','FontSize',14);
title(['Phasenportrait: \dot{I} = I - K\,S,  \dot{S} = I - C\,S - G_0'], ...
      'Interpreter','tex','FontSize',16,'FontWeight','bold');

grid on; box on;
legend({'Vektorfeld','Trajektorien'}, 'Location','northeast');
