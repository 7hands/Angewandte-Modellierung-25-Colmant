%numeric calc
syms omega

H(omega) = 1/(1+ 1j*omega*0.5*(1492e-6)); %Funktion

amp(omega) = abs(H(omega));

amp_num = matlabFunction(amp);

amp(0)

phase = matlabFunction(angle(H));

phase(1200)


figure
fplot(amp(omega), [-120, 120]);           % Betrag (Amplitude)
figure
fplot(angle(H(omega)), [-120, 120]);         % Phase

