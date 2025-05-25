% wave_fd.m
% Löse d u_tt = c * u_xx + f − a*u mit d=1, c=1, a=0, f=0
% auf [0,1], u(0,t)=u(1,t)=0,
% Anfang: u(x,0)=sin(2*pi*x), ut(x,0)=0

clear; close all; clc;

%% Parameter
c    = 1;      % Wellen­geschwindigkeit^2
L    = 1;      % Länge des Intervalls
Tend = 1;      % Endzeit
Nx   = 101;    % Gitterpunkte in x
dx   = L/(Nx-1);
CFL  = 0.9;    % CFL-Zahl <1 für Stabilität
dt   = CFL*dx/sqrt(c);
Nt   = floor(Tend/dt);
dt   = Tend/Nt;   % präzisieren, damit Nt*dt=Tend

x = linspace(0,L,Nx)';
t = (0:Nt)*dt;

%% Initialbedingungen
U = zeros(Nt+1, Nx);
U(1,:) = sin(2*pi*x)';      % u(x,0)
% ut(x,0)=0 => erster Zeitschritt mit Taylor-Ansatz
r2 = c*dt^2/dx^2;
U(2, 2:end-1) = U(1,2:end-1) + ...
    0.5*r2*(U(1,3:end) - 2*U(1,2:end-1) + U(1,1:end-2));
% Randwerte bleiben null:
U(2,1) = 0;  
U(2,end) = 0;

%% Zeitintegration (explizit)
for n = 2:Nt
    U(n+1,2:end-1) = 2*U(n,2:end-1) - U(n-1,2:end-1) + ...
        r2*(U(n,3:end) - 2*U(n,2:end-1) + U(n,1:end-2));
    % Dirichlet-BC 
    U(n+1,1) = 0;
    U(n+1,end) = 0;
end

%% 1) Oberflächenplot
[X,T] = meshgrid(x, t);
figure;
surf(X, T, U, 'EdgeColor', 'none');
xlabel('x'); ylabel('t'); zlabel('u(x,t)');
title('Wellen­gleichung: u_{tt} = c u_{xx}');
view(45,30);
colorbar;
shading interp;

%% 2) Animation
figure;
h = plot(x, U(1,:), 'LineWidth', 2);
axis([0 1 -1 1]);
xlabel('x'); ylabel('u');
title(sprintf('t = %.3f',0));
drawnow;

for n = 1:Nt+1
    set(h, 'YData', U(n,:));
    title(sprintf('Wellen­ausbreitung bei t = %.3f', t(n)));
    pause(0.02);
end
