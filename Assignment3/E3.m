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