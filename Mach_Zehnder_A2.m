%% Simulación del interferómetro de MachZehnder
clc; clear; close all;
% Parámetros del haz gaussiano
lambda = 632.8e-9; % Longitud de onda en metros (Láser He-Ne)
k = 2*pi/lambda; % Número de onda
w0 = 1e-3; % Ancho del haz en metros
z0 = pi*w0^2/lambda; % Distancia de Rayleigh
N = 500;

% Parámetros de la rejilla espacial
x = linspace(-2*w0, 2*w0, N);
y = linspace(-2*w0, 2*w0, N);
[X, Y] = meshgrid(x, y);

% Parámetros de los haces
z = 0; % Plano de observación
theta = pi/16; % Ángulo entre los dos haces

% Campos eléctricos iniciales
E1 = exp(-(X.^2 + Y.^2) / w0^2).*exp(1i*(k*z + k*X*sin(theta)));
E2 = exp(-(X.^2 + Y.^2) / w0^2).*exp(1i*(k*z - k*X*sin(theta)));

% Corrimiento de fase variable
% Fase dependiente de y para la mitad inferior
n_center = 1;
n_bottom = 4;
d = 1; % Distancia de la placa

% Calcular el cambio de índice de refracción cada 30 filas
rows_per_step = 8;
num_steps = floor(N / (2 * rows_per_step));
delta_n = (n_center - n_bottom) / num_steps;

n = n_center * ones(N / 2, 1);
for i = 1:num_steps
    n((i-1)*rows_per_step + 1 : i*rows_per_step) = rand*n_center - rand*delta_n * (i - 1);
end
n = [n; ones(N / 2, 1)]; % Para la mitad superior, n = 1

% Crear la matriz de cambio de fase
phase_shift = k * d * n - 1 .* (Y <= 0);

% Matrices de transferencia
BS = 1/sqrt(2) * [1, 1i; 1i, 1];                % Divisor de haz 50-50
Mirror = exp(1i*pi);                            % Espejo plano
ES = @(d) exp(1i * k * d);                      % Propagación en el espacio libre
V = @(p) exp(1i*p);                             % Placa de vidrio
L = @(f) exp(1i*((k/(2*f))*(X.^2 + Y.^2)));     % Lente delgada
F = exp(1i * phase_shift);                      % Flama

% Camino 1: Propagación -> Espejo -> Propagación -> Divisor de haz
% Aplicar propagación en el espacio libre (d1)
d1 = 1;
E1 = E1 .* ES(d1);
% Aplicar reflejo en espejo
E1 = E1 .* Mirror;
% Segunda propagación
E1 = E1 .* ES(d1);

% Camino 2: Propagación -> Espejo -> Propagación -> Divisor de haz
% Aplicar propagación en el espacio libre (d2)
d1 = 1;
E2 = E2 .* ES(d1);
% Aplicar reflejo en espejo
E2 = E2 .* Mirror;
% Segunda propagación
E2 = E2 .* ES(d1);

% Casos adicionales
% Placa de vidrio
% E2(N/2:end,:) = V(pi/2) .* E2(N/2:end,:);
% Lente delgada
% E2 = L(0.08) .* E2;
% Flama
E2 = F .* E2;

% Aplicar el divisor de haz de salida
Output1 = BS(1,1) * E1 + BS(1,2) * E2;
Output2 = BS(2,1) * E1 + BS(2,2) * E2;

% Intensidades en las salidas
I1 = abs(Output1).^2;
I2 = abs(Output2).^2;

Imax1 = max(max(I1));
Imax2 = max(max(I2));

z_line = ones(size(x)) * 1;

% Graficar las intensidades
figure;
surf(X,Y,I1./Imax1);
shading interp;
colormap('hot');
colorbar;
xlim([-1.5*w0 1.5*w0]);
ylim([-1.5*w0 1.5*w0]);
xlabel('$X$ (m)','Interpreter','latex','FontSize',20);
ylabel('$Y$ (m)','Interpreter','latex','FontSize',20);
view(2)
% hold on
% plot3(x(end/2)*ones(size(x)), x, z_line, 'r', 'LineWidth', 2);
% hold off

figure;
surf(X,Y,I2./Imax2);
shading interp;
colormap('bone');
colorbar;
xlim([-1.5*w0 1.5*w0]);
ylim([-1.5*w0 1.5*w0]);
xlabel('$X$ (m)','Interpreter','latex','FontSize',20);
ylabel('$Y$ (m)','Interpreter','latex','FontSize',20);
view(2)
% hold on
% plot3(x(end/2)*ones(size(x)), x, z_line, 'r', 'LineWidth', 2);
% hold off
%% Perfiles transversales experimentales.
clc; clear; close all;

S1 = double(rgb2gray(imread('S1.JPG')));
S2 = double(rgb2gray(imread('S2.JPG')));
S3 = double(rgb2gray(imread('S3.JPG')));
S4 = double(rgb2gray(imread('S4.JPG')));
S5 = double(rgb2gray(imread('S5.JPG')));

% figure;
% imagesc(S5)

figure;
subplot(2,1,1)
x1 = 1500:1:5000;
PS1 = S1(1200,1500:5000);
plot(x1, PS1/max(PS1),'b');
title('Puerto de salida 1','Interpreter','latex','FontSize', 16)
xlabel('\''Indice de columnas','Interpreter','latex','FontSize', 16)
ylabel('$I$ normalizada','Interpreter','latex','FontSize', 16)
grid on
subplot(2,1,2)
x2 = 1000:1:4500;
PS2 = S2(2200,1000:4500);
plot(x2, PS2/max(PS2),'r');
title('Puerto de salida 2','Interpreter','latex','FontSize', 16)
xlabel('\''Indice de columnas','Interpreter','latex','FontSize', 16)
ylabel('$I$ normalizada','Interpreter','latex','FontSize', 16)
grid on

% I0 = max(PS1);
x3 = 1500:1:4500;
H = 1000;
PS4 = S4(H,1500:4500);
PS5 = S5(H,1500:4500);

figure;
subplot(2,1,1)
plot(x3, PS4/max(PS4),'Color','#77AC30');
title('Filtro en posici\''on A','Interpreter','latex','FontSize', 16)
xlabel('\''Indice de columnas','Interpreter','latex','FontSize', 16)
ylabel('$I$ normalizada','Interpreter','latex','FontSize', 16)
grid on
subplot(2,1,2)
plot(x3, PS5/max(PS5),'Color','#7E2F8E');
title('Filtro a la salida','Interpreter','latex','FontSize', 16)
xlabel('\''Indice de columnas','Interpreter','latex','FontSize', 16)
ylabel('$I$ normalizada','Interpreter','latex','FontSize', 16)
grid on

%% Datos suavizados con redes neuronales
close all; clc

% Prepara los datos de entrada y objetivo
input = x3;
target = PS4/max(PS4);

% Define la arquitectura de la red neuronal
hiddenLayerSize = [10 10];  % Cambia la arquitectura según sea necesario
net = feedforwardnet(hiddenLayerSize,'trainbr');  % Prueba diferentes funciones de entrenamiento

% Entrena la red neuronal con validación cruzada
net.divideFcn = 'dividerand';  % Cambia la función de división
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Entrena la red neuronal
[net,tr] = train(net, input, target);

% Utiliza la red entrenada para predecir los datos suavizados
predictedOutputs = net(input);

% Visualización y análisis de resultados
smoothedData = predictedOutputs(1, :);
[maxima, maxIndices] = findpeaks(smoothedData);
[minima, minIndices] = findpeaks(-smoothedData);
minima = -minima;

figure;
plot(x3, smoothedData, 'Color','#77AC30');
hold on;
% plot(x1, target, 'r');
plot(x3(maxIndices), maxima, 'r*', 'MarkerSize', 10);
plot(x3(minIndices), minima, 'b*', 'MarkerSize', 10);
title('Filtro en posici\''on A','Interpreter','latex','FontSize', 11)
xlabel('\''Indice de columnas','Interpreter','latex','FontSize', 11)
ylabel('$I$ normalizada','Interpreter','latex','FontSize', 11)
grid on;
hold off;
lgd = legend('Suavizado', 'M\''aximos', 'M\''inimos', 'Interpreter', 'latex');
lgd.Location = 'best';

disp(max(maxima))
disp(min(minima))
