% diffusion_implicit.m
% Löse du/dt = D * d2u/dx2 auf [0,1] mit u(0)=u(1)=0,
% Anfangsbedingung Gauß bei x=0.5, D=0.01, implizites Euler + Finite Volume

clear; close all; clc;

%% Parameter
D     = 0.01;      % Diffusionskoeffizient
L     = 1;         % Länge des Stabes
Tmax  = 0.2;       % Endzeit
Nx    = 51;        % Anzahl Gitterpunkte in x
Nt    = 200;       % Anzahl Zeitschritte

dx = L/(Nx-1);
dt = Tmax/Nt;
x  = linspace(0,L,Nx)';
t  = linspace(0,Tmax,Nt+1);

%% Anfangsbedingung
u0 = exp(-100*(x-0.5).^2);

% Matrix A für implizites Euler (innere Punkte 2..Nx-1)
N     = Nx-2;
alpha = D*dt/dx^2;
main  = (1 + 2*alpha) * ones(N,1);
off   = -alpha       * ones(N-1,1);
col_im1 = [0;        off];
col_ip1 = [off;      0 ];
A     = spdiags([col_im1, main, col_ip1], -1:1, N, N);


%% Zeitintegration
U = zeros(Nx, Nt+1);
U(:,1) = u0;

for n = 1:Nt
    b = U(2:end-1,n);         % RHS = alte Lösung innen
    U(2:end-1,n+1) = A\b;     % löse A * u^{n+1} = b
    % U(1,n+1)=0; U(end,n+1)=0; % BCs sind schon null initialisiert
end

%% 1) Oberflächenplot
[X,T] = meshgrid(x,t);  % Achtung: T(i,j) = t(i), X(i,j)=x(j)
figure;
surf(X, T, U', 'EdgeColor','none');
xlabel('x'); ylabel('t'); zlabel('u(x,t)');
title('Surface plot of u(x,t)');
view(45,30);
colorbar;
shading interp;

%% 2) Animation
figure;
h = plot(x, U(:,1), 'LineWidth',2);
axis([0 1 0 1]);
xlabel('x'); ylabel('u');
title(sprintf('t = %.3f',0));
drawnow;

for n = 1:Nt+1
    set(h, 'YData', U(:,n));
    title(sprintf('Diffusion t = %.3f', t(n)));
    pause(0.05);
end
