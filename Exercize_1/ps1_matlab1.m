% Introduction to Matlab
% Part 1
% Run these commands in the Command Window. Use copy and paste.
% Correct all errors and save your code

clear                               % Remove items from workspace.
3^2*4-3*2^5*(4-2);                   % Simple arithmetic.
sqrt(25)                            % Square root.
x=1:2:9;                             % A vector.
y=x.^2;                              % Square the elements.
A=[1,2; 3,4];                         % A 2x2 matrix.
A';                            % The transpose.
det(A)                              % The determinant.
B=[0,3,1; .3,0,0; 0,.5,0];             % A 3x3 matrix.
eig(B)                              % The eigenvalues of B.
[Vects,Vals]=eig(B)                 % Eigenvectors and eigenvalues.
C=[100; 200; 300]                     % A 3x1 matrix.
D=B*C                               % Matrix multiplication.
E=B^4                               % Powers of matrices.
z1=1+1i                              % Complex numbers.
z2=1-1i
z3=2+1i
z4=2*z1-z2*z3                       % Complex arithmetic.
abs(z1)                             % Modulus.
real(z1)                            % Real part.
imag(z1)                            % Imaginary part.
exp(1i*z1)                           % Exponential.

sym(1/2)+sym(3/4)                   % Symbolic arithmetic.
1/2+3/4                             % Double precision.
vpa(pi,100)                         % Variable precision.
help vpa                            % Shows help

% Symbolic calculation
syms x(t) y(t) z(t) t               % Symbolic objects
z=x^3-y^3                           % additional description
factor(z)                           % Factorization.
expand(6*cos(t-pi/4))               % Expansion.
simplify(z/(x-y))                   % Simplification.
limit(x/sin(x),x,0)                 % Limits.
clear
clc


syms x y
[x,y]=solve(x^2-x==0,2*x*y-y^2==0);  % Solving simultaneous equations.
syms x mu
f=mu*x^2*(1-x)                      % Define a function.
subs(f,x,1/3)                       % Evaluate f(1/3).
fof=subs(f,x,f)                     % Composite function.
diff(f,x)                           % Differentiation.

syms x y
diff(x^2+3*x*y-2*y^2,y,2)           % Partial differentiation.
int(sin(x)*cos(x),x,0,pi/2)         % Integration.
int(1/x,x,1,inf)                    % Improper integration.

syms n s w
s1=symsum(1/n^2,1,inf)              % Symbolic summation.
g=exp(x)
taylor(g,'Order',10)                % Taylor series up to order 10.

syms a w
laplace(x^3)                        % Laplace transform.
ilaplace(1/(s-a))                   % Inverse transform.
fourier(exp(-x^2))                  % Fourier transform.
ifourier(pi/(1+w^2))                % Inverse transform.

% End of Part 1.


% Part 2
% Graph and simple differential equations

clear
% Plot a simple function.
x=-2:.01:2;
plot(x,x.^2)                            

% Plot two functions on one graph.
t=0:.1:100;
y1=exp(-.1*t).*cos(t);y2=cos(t);
plot(t,y1,t,y2),legend('y1','y2')       

% Symbolic plots.
fplot(@(x) x.^2,[-2,2])
fun = (@(t) exp(-1.*t).*sin(t));
fplot(fun),xlabel('time'),ylabel('current'),title('decay')

% 3-D plots on a 50x50 grid.

f = @(x,y) sin(x) + cos(y);
fcontour(f)
fcontour(@(x,y) sin(x) + cos(y))
f1 = @(x,y) y.^2/2-x.^2/2+x.^4/4;
fcontour(f1,[-2 2],'MeshDensity',50)
fsurf(f1,[-2 2],'MeshDensity',50)
fsurf(f1,[-3 3],'ShowContours','on')

% Parametric plot.
ezplot('t^3-4*t','t^2',[-3,3])
fplot(@(t) t.^2+4*t,@(t) t.^2,[-5,2])

% 3-D parametric plot.
ezplot3('sin(t)','cos(t)','t',[-10,10])

% Symbolic solutions to ODEs.
dsolve('Dx=-x/t')
dsolve('D2x+5*Dx+6*x=10*sin(t)','x(0)=0','Dx(0)=0')
clear
clc
syms x(t)
dsolve(diff(x,t) == -x/t, x(1) == 1)
dsolve(diff(x,t) == -t/x, x(0) == 1)

% Linear systems of ODEs.
syms x(t) y(t)
eqns = [diff(x,t) == 3*x+4*y, diff(y,t) == -4*x+3*y];
S = dsolve(eqns);
ySol(t) = S.y;

[x,y]=dsolve('Dx=x^2','Dy=y^2','x(0)=1,y(0)=1');

% A 3-D linear system.
syms x(t) y(t) z(t) t
[x,y,z]=dsolve('Dx=x','Dy=y','Dz=-z');
[x,y,z]=dsolve(diff(x,t)==x,diff(y,t)==y,diff(z,t)==z);

% Numerical solutions to ODEs.
% You put your comments here
deq1=@(t,x) x(1)*(.1-.01*x(1));
[t,xa]=ode45(deq1,[0 100],50);
plot(t,xa(:,1))

% A 2-D system.
deq2=@(t,x) [.1*x(1)+x(2);-x(1)+.1*x(2)];
[t,xb]=ode45(deq2,[0 50],[.01,0]);
plot(xb(:,1),xb(:,2))

% A 3-D system.
deq3=@(t,x) [x(3)-x(1);-x(2);x(3)-17*x(1)+16];
[t,xc]=ode45(deq3,[0 20],[.8,.8,.8]);
plot3(xc(:,1),xc(:,2),xc(:,3))

% A stiff system.
deq4=@(t,x) [x(2);1000*(1-(x(1))^2)*x(2)-x(1)];
[t,xd]=ode23s(deq4,[0 3000],[.01,0]);
plot(xd(:,1),xd(:,2))

% x versus t.
plot(t,xd(:,1))

% End of Part 2.