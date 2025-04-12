% Parameter
Fs = 44100;                      % Abtastrate (z. B. 44100 Hz)
y = out.simout;                   % Dein Audiosignal (Vektor)
N = length(y);                   % Anzahl Samples

% Frequenzspektrum berechnen
Y = fft(y);                      % FFT berechnen
f = (0:N-1)*(Fs/N);              % Frequenzachse
P = abs(Y)/N;                    % Amplitudenspektrum normalisiert

% Nur bis zur Nyquist-Frequenz anzeigen
half = 1:floor(N/2);

% Plot
plot(f(half), P(half));
xlabel('Frequenz (Hz)');
ylabel('Amplitude');
title('Frequenzspektrum');
grid on;
