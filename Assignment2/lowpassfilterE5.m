%numeric calc
syms omega

H(omega) = 1/(1+ 1j*omega*0.5*(1492e-6)); %Funktion

amp(omega) = abs(H(omega));

amp_num = matlabFunction(amp); %turn symbolick funktion into numeric one

amp(0); % Amplitude at omega = 0

phase = matlabFunction(angle(H));

phase(0) % Phase at omega = 0

%plot
figure
fplot(amp(omega), [-120, 120]);           % Betrag (Amplitude)
figure
fplot(angle(H(omega)), [-120, 120]);         % Phase

