%the first null in the Envelope is at 80kHz
% f_0 = 2kHz
% T = 1/f_0

T = 1/10000;

% for the Pulse width:
% f = 1/roh
% f = 80kHz
roh = 1/80000;

%period width percent = rho/period *100

w = roh/T * 100


%for pictures for E4

% wir lassen w von -25000 bis 25000 mit 50er schritten gehen.
for w=-25000:50:25000 

    % Das ist an sich eine Fourier Transformation nur das das integral
    % durch eine Riemann summe ausgetauscht wurde/approximiert wird. Die
    % rectpuls funktion agiert hier als ein Ersatz für den dirac Impuls der
    % nur Mathematisch existiert. Die breite des rectpuls entscheidet über 
    % die genauig keit der Fourier Transformation.
    Fw=sum(rectpuls(t_vector, 0.025).*exp(-1i*w*t_vector))*t_step;
    
    %Die Werte werden gespeichert damit sie nach dem For loop geplottet
    %werden können.
    w_vector=[w_vector w];
    Fw_vector=[Fw_vector Fw];
    
end
plot(w_vector,abs(Fw_vector))
xlabel('Frequency [rad/sec]')
grid



clear
load for_ps3.mat
t_step = t_vector(2) - t_vector(1);

% Länge des ft Vectors
L = length(ft_vector);

% Berechung der Ableitung von den Werten des ft_vectors
diff_ft = (ft_vector(2:L) - ft_vector(1:(L-1))) / t_step;

% Den Zeit vector (t_vector) an die Länge der Ableitung anpassen weil wir
% die Ableitung numerisch ausgerechnet haben zwischen zwei wert Paren fehlt
% hinten ein wert.
t_vector = t_vector(1:(L-1));

Dw_vector = [];
w_vector = [];

% wir lassen w von -25000 bis 25000 mit 50er schritten gehen.
for w = -25000:50:25000
    

    % Die Fourier Transformation für die Ableitung ausrechnen.
    % die Fehlenden Terme waren diff_ft, w und der neue t_vector. Die
    % Ableitung wird wie eine Art Fenster benutzt.
    Dw = sum(diff_ft .* exp(-j * w * t_vector)) * t_step;

    % Den Wert der Ableitung dem DW_Vector hinzufügen
    Dw_vector = [Dw_vector Dw];
    % w wert zum w vector hinzufügen.
    w_vector = [w_vector w];
end

% Plot
plot(w_vector, abs(Dw_vector));
xlabel('Frequency [rad/sec]');
grid on;

