% MATLAB-Skript zur Erstellung eines diskreten Simulink-Modells "spectrum_analysis_model"
% Modell: diskreter Sinus + Rauschen + FIR Lowpass Filter + Buffer + Spectrum Analyzer

model = 'spectrum_analysis_model';
% Schließe falls offen\ nif bdIsLoaded(model)
    close_system(model, 0);

% Neues Modell erstellen
new_system(model);
open_system(model);

% Parameter
ts = 1/1000;        % Abtastrate 1000 Hz
fftLen = 1024;      % FFT-Länge für hohe Frequenzauflösung
fc = 100;           % Filter-Cutoff 100 Hz

% FIR-Filterkoeffizienten erstellen (Order 48, Hamming)
nFilt = 48;
b = fir1(nFilt, fc/(1/(2*ts)), 'low', hamming(nFilt+1));

% Positionierung
x0 = 30; y0 = 30;

% 1) Discrete Sine Wave (50 Hz)
add_block('simulink/Sources/Sine Wave', [model '/Sine Wave'], ...
    'Position', [x0 y0 x0+100 y0+50], ...
    'Frequency', '50', 'Amplitude', '1', 'SampleTime', num2str(ts));

% 2) Random Number (Rauschen)
add_block('simulink/Sources/Random Number', [model '/Noise'], ...
    'Position', [x0 y0+100 x0+100 y0+150], ...
    'Mean', '0', 'Variance', '0.2', 'SampleTime', num2str(ts));

% 3) Sum
add_block('simulink/Math Operations/Add', [model '/Sum'], ...
    'Position', [x0+200 y0+50 x0+260 y0+100], 'Inputs', '|++');



% 5) Rate Transition
add_block('simulink/Signal Attributes/Rate Transition', [model '/Rate Transition'], ...
    'Position', [x0+500 y0+50 x0+550 y0+100]);

% 6) Buffer
add_block('dspsigops/Buffer', [model '/Buffer'], ...
    'Position', [x0+600 y0+50 x0+680 y0+100], ...
    'BufferSize', num2str(fftLen), 'Overlap', '0', 'OutputAsMatrix', 'off');

% 7) Spectrum Analyzer
add_block('dsp/Spectrum Analyzer', [model '/Spectrum Analyzer'], ...
    'Position', [x0+760 y0+30 x0+920 y0+200], ...
    'SampleRate', '1000', 'PlotAsTwoSidedSpectrum', 'on', ...
    'FrequencySpan', 'Full', 'FFTLengthSource', 'Property', 'FFTLength', num2str(fftLen), ...
    'Title', 'Spektrumanalyse');

% Verbindungen
add_line(model, 'Sine Wave/1', 'Sum/1');
add_line(model, 'Noise/1', 'Sum/2');
add_line(model, 'Sum/1', 'FIR Lowpass/1');
add_line(model, 'FIR Lowpass/1', 'Rate Transition/1');
add_line(model, 'Rate Transition/1', 'Buffer/1');
add_line(model, 'Buffer/1', 'Spectrum Analyzer/1');

% Speicher & offen
save_system(model);
open_system([model '/Spectrum Analyzer']);

fprintf('Modell "%s" erstellt. Simulation starten (Ctrl+T) für Peak bei 50 Hz.\n', model);
