% Eric Fernández García / A00833613
% Polarización y parámetros de Stokes
clc; clear
%% PARTE 1: Generación del campo óptico
close all
% Parámetros
w0 = 1; % Cintura del haz
m = 2;  % Carga topológica

% Resolución de la ventana numérica
N = 256; % Número de puntos (potencia de 2)

r_max = 3 * w0;
phi_max = 2 * pi;
r = linspace(0, r_max, N);
phi = linspace(0, phi_max, N);
[R, PHI] = meshgrid(r, phi);

% Campo óptico en coordenadas polares
E = (R/w0).^abs(m) .* exp(-R.^2 / w0^2) .* exp(1i * m * PHI);
Imax = max(max(abs(E).^2));
[X, Y] = pol2cart(PHI, R);

figure;
colormap hot;
surf(X, Y, (abs(E).^2)./Imax);
shading interp;
title('Campo \''optico $\left|E\left(r, \phi\right)\right|^2$','Interpreter','latex');
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
zlabel('$\left|E\right|^2$','Interpreter','latex');
colorbar;
view(2);
axis equal;
%% PARTE 2: Estado de polarización inicial del campo óptico
close all
% Estado de polarización
state = 'V'; % Cambiar según el estado deseado ('H', 'V', 'RC', 'LC', 'D', 'A', 'E', 'L')
[Ex, Ey] = polarization_state(state);
% [Ex, Ey] = arbitrary_polarization(state, pi/8, pi/8);

% Campo óptico con estado de polarización inicial
E_x = E * Ex;
E_y = E * Ey;
I0 = sum(sum(abs(E_x).^2 + abs(E_y).^2));       % Intensidad inicial

% Representación gráfica del campo con polarización
figure('Position', [200, 200, 800, 400]);
colormap hot;
subplot(1, 2, 1);
surf(X, Y, (abs(E_x).^2)./Imax);
shading interp;
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
title(['$\left|E_x\left(r, \phi\right)\right|^2$ (Estado: ', state, ')'],'Interpreter','latex');
colorbar;
view(2)
subplot(1, 2, 2);
surf(X, Y, (abs(E_y).^2)./Imax);
shading interp;
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
title(['$\left|E_y\left(r, \phi\right)\right|^2$ (Estado: ', state, ')'],'Interpreter','latex');
colorbar;
view(2)
%% PARTE 3: Polarizador lineal a cualquier ángulo
close all;
D = 37;
Ip = zeros(1, D);
% P = zeros(1, D);
theta_values = linspace(0, 2*pi, D);

for i = 1:length(theta_values)
    theta = theta_values(i);
    [Ex2, Ey2] = polarization_angle(theta, Ex, Ey);

    % Campo óptico resultante del polarizador lineal
    E_xp = E * Ex2;
    E_yp = E * Ey2;

    Ip(i) = sum(sum(abs(E_xp).^2 + abs(E_yp).^2));      % Intensidad
%     P(i) = trapz(trapz(abs(E_xp).^2 + + abs(E_yp).^2)); % Potencia
end

Malus = abs(Ex * cos(theta_values) + Ey * sin(theta_values)).^2;
Ip_normalized = Ip / I0;

% Nombre del archivo Excel
archivo_excel = 'DATA.xlsx';

% Leer datos de la columna A
data_A = xlsread(archivo_excel, 'A:A');

MAE = (1/length(data_A))*sum(abs(data_A' - Ip_normalized));
disp(MAE)
MAPE = (1/length(data_A))*sum(abs((data_A' - Ip_normalized)/Ip_normalized))*100;
disp(MAPE)

figure; 
subplot(2, 1, 1);
plot(theta_values, Ip_normalized, 'LineWidth', 1, 'Color', 'r');
xlabel('$\theta$ (Polarizador lineal)', 'Interpreter', 'latex');
ylabel('$I\left(\theta\right)$', 'Interpreter', 'latex');
title('Simulaci\''on num\''erica', 'Interpreter', 'latex');
xlim([theta_values(1) theta_values(end)]);
xticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
grid on;
subplot(2, 1, 2);
plot(theta_values, Malus, 'LineWidth', 1, 'Color', 'b');
xlabel('$\theta$ (Polarizador lineal)', 'Interpreter', 'latex');
ylabel('$I\left(\theta\right)$', 'Interpreter', 'latex');
title('Te\''orico (Ley de Malus)', 'Interpreter', 'latex');
xlim([theta_values(1) theta_values(end)]);
xticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
grid on;
sgtitle('Intensidad vs. \''Angulo ($\theta$ en radianes)', 'Interpreter', 'latex');


figure; 
plot(theta_values, Ip_normalized, 'LineWidth', 1, 'Color', 'k', 'LineStyle', '--');
hold on
plot(theta_values, data_A', 'LineWidth', 1, 'Color', 'g');
xlabel('$\theta$ (Polarizador lineal)', 'Interpreter', 'latex');
ylabel('$I\left(\theta\right)$', 'Interpreter', 'latex');
title('Intensidad vs. \''Angulo ($\theta$ en radianes)', 'Interpreter', 'latex')
xlim([theta_values(1) theta_values(end)]);
xticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
legend('Curva te\''orica (Ley de Malus)', 'Curva experimental', 'Interpreter', 'latex')
grid on;
%% PARTE 4: Retardador λ/4 y polarizador lineal a cualquier ángulo
close all;
D = 160;
Ip = zeros(D, D);
theta_values = linspace(0, 2*pi, D);
omega_values = linspace(0, 2*pi, D);

for i = 1:length(theta_values)
    theta = theta_values(i);
    [Ex2, Ey2] = retarder4(theta, Ex, Ey);
    for j = 1:length(theta_values)
        omega = omega_values(j);
        [Ex3, Ey3] = polarization_angle(omega, Ex2, Ey2);
        E_xp = E * Ex3;
        E_yp = E * Ey3;
        % Campo óptico resultante del polarizador lineal y retardador λ/4
        Ip(i,j) = sum(sum(abs(E_xp).^2 + abs(E_yp).^2));
    end
end

Ip_normalized = Ip / I0;

figure;
colormap hot;
surf(omega_values, theta_values, Ip_normalized);
ylabel('$\theta$ (Retardador $\lambda/4$)', 'Interpreter', 'latex');
xlabel('$\omega$ (Polarizador lineal)', 'Interpreter', 'latex');
zlabel('$I(\theta, \alpha)$', 'Interpreter', 'latex');
title('Intensidad vs. \''Angulos ($\theta$ y $\omega$ en radianes)', 'Interpreter', 'latex');
xlim([theta_values(1) theta_values(end)]);
ylim([omega_values(1) omega_values(end)]);
grid on;
colorbar;
shading interp;
view(2)

z_line = ones(size(omega_values)) * 1;

hold on
plot3(omega_values, theta_values(20)*ones(size(omega_values)), z_line, 'k', 'LineWidth', 1);
plot3(omega_values, theta_values(60)*ones(size(omega_values)), z_line, 'k', 'LineWidth', 1);
plot3(omega_values, theta_values(100)*ones(size(omega_values)), z_line, 'k', 'LineWidth', 1);
plot3(omega_values, theta_values(140)*ones(size(omega_values)), z_line, 'k', 'LineWidth', 1);
hold off

xticks(0:pi/4:2*pi);
yticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
yticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
%% PARTE 5: Retardador λ/2 y polarizador lineal a cualquier ángulo
close all;
D = 100;
Ip = zeros(D, D);
theta_values = linspace(0, 2*pi, D);
omega_values = linspace(0, 2*pi, D);

for i = 1:length(theta_values)
    theta = theta_values(i);
    [Ex2, Ey2] = retarder2(theta, Ex, Ey);
    for j = 1:length(theta_values)
        omega = omega_values(j);
        [Ex3, Ey3] = polarization_angle(omega, Ex2, Ey2);
        E_xp = E * Ex3;
        E_yp = E * Ey3;
        % Campo óptico resultante del polarizador lineal y retardador λ/2
        Ip(i,j) = sum(sum(abs(E_xp).^2 + abs(E_yp).^2));
    end
end

Ip_normalized = Ip / I0;

figure;
colormap hot;
surf(omega_values, theta_values, Ip_normalized);
ylabel('$\theta$ (Retardador $\lambda/2$)', 'Interpreter', 'latex');
xlabel('$\omega$ (Polarizador lineal)', 'Interpreter', 'latex');
zlabel('$I(\theta, \alpha)$', 'Interpreter', 'latex');
title('Intensidad vs. \''Angulos $\left(\theta, \omega\right)$', 'Interpreter', 'latex');
xlim([theta_values(1) theta_values(end)]);
ylim([omega_values(1) omega_values(end)]);
grid on;
colorbar;
shading interp;
view(2)

xticks(0:pi/4:2*pi);
yticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
yticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
%% PARTE 6: Parámetros de Stokes con retardador λ/2 a cualquier ángulo
close all;
D = 200;
Ip = zeros(1, D);
S0 = zeros(1, D);
S1 = zeros(1, D);
S2 = zeros(1, D);
S3 = zeros(1, D);
P = zeros(1, D);
omega_values = linspace(0, 2*pi, D);

for i = 1:length(omega_values)
    omega = omega_values(i);
    [Ex3, Ey3] = retarder4(omega, Ex, Ey);

    % Campo óptico resultante del retardador λ/2
    E_xp = E * Ex3;
    E_yp = E * Ey3;

    Ip(i) = sum(sum(abs(E_xp).^2 + abs(E_yp).^2));      % Intensidad
    [S0(i), S1(i), S2(i), S3(i), P(i)] = Stokes(E_xp, E_yp, I0);
end

Ip_normalized = Ip / I0;

figure;
hold on
plot(omega_values, S0, 'LineWidth', 1, 'Color', 'r');
plot(omega_values, S1, 'LineWidth', 1, 'Color', 'b');
plot(omega_values, S2, 'LineWidth', 1, 'Color', '#77AC30');
plot(omega_values, S3, 'LineWidth', 1, 'Color', '#D95319');
plot(omega_values, P, 'LineWidth', 1, 'Color', '#7E2F8E');
hold off
legend('$S_0$','$S_1$','$S_2$','$S_3$','$P$', 'Interpreter', 'latex','Location','bestoutside')
xlabel('$\omega$ (Retardador $\lambda/2$)', 'Interpreter', 'latex');
ylabel('Coeficiente', 'Interpreter', 'latex');
title('Par\''ametros de Stokes vs. \''Angulo ($\omega$ en radianes)', 'Interpreter', 'latex');
xlim([omega_values(1) omega_values(end)]);
xticks(0:pi/4:2*pi);
xticklabels({'0', '$\pi/4$', '$\pi/2$', '$3\pi/4$', '$\pi$', '$5\pi/4$', '$3\pi/2$', '$7\pi/4$', '$2\pi$'});
set(gca, 'TickLabelInterpreter', 'latex');
grid on;
%% Análisis individual con los parámetros de Stokes
[S0, S1, S2, S3, P] = Stokes(E_x, E_y, I0);
fprintf('S0 = %.4f\n', S0);
fprintf('S1 = %.4f\n', S1);
fprintf('S2 = %.4f\n', S2);
fprintf('S3 = %.4f\n', S3);
fprintf('P = %.4f\n', P);
%% Función de estado de polarización
function [Ex, Ey] = polarization_state(state)
    switch state
        case 'H'                                % Horizontal
            Ex = 1; Ey = 0;
        case 'V'
            Ex = 0; Ey = 1;                     % Vertical
        case 'RC'
            Ex = 1/sqrt(2); Ey = 1i/sqrt(2);    % Circular derecho
        case 'LC'
            Ex = 1/sqrt(2); Ey = -1i/sqrt(2);   % Circular izquierdo
        case 'D'
            Ex = 1/sqrt(2); Ey = 1/sqrt(2);     % Diagonal
        case 'A'
            Ex = 1/sqrt(2); Ey = -1/sqrt(2);    % Antidiagonal
        otherwise
            error('Estado de polarización no reconocido');
    end
end
%% Función de estado de polarización arbitrario (esfera de Poincaré)
function [Ex, Ey] = arbitrary_polarization(state, ye, ve)
    switch state
        case 'L'        % Lineal
            Ex = cos(ye); Ey = sin(ye);
        case 'E'        % Elíptica
            Ex = cos(ye)*cos(ve) + 1i*sin(ye)*sin(ve); Ey = sin(ye)*cos(ve) - 1i*cos(ye)*sin(ve);
        otherwise
            error('Estado de polarización no reconocido');
    end
end
%% Polarizador lineal a cualquier ángulo
function [Ex, Ey] = polarization_angle(theta, Ex, Ey)
LP = [cos(theta)^2, cos(theta)*sin(theta); cos(theta)*sin(theta), sin(theta)^2];
NE = LP*[Ex; Ey];
Ex = NE(1);
Ey = NE(2);
end
%% Retardador de cuarto de onda a cualquier ángulo
function [Ex, Ey] = retarder4(theta, Ex, Ey)
LP = (1/sqrt(2))*[1 - 1i*cos(2*theta), -2i*cos(theta)*sin(theta); -2i*cos(theta)*sin(theta), 1 + 1i*cos(2*theta)];
NE = LP*[Ex; Ey];
Ex = NE(1);
Ey = NE(2);
end
%% Retardador de media onda a cualquier ángulo
function [Ex, Ey] = retarder2(theta, Ex, Ey)
LP = [-1i*cos(2*theta), -1i*sin(2*theta); -1i*sin(2*theta), 1i*cos(2*theta)];
NE = LP*[Ex; Ey];
Ex = NE(1);
Ey = NE(2);
end
%% Parámetros de Stokes y grado de polarización
function [S0, S1, S2, S3, P] = Stokes(E_x, E_y, I0)
S0 = sum(sum(abs(E_x).^2 + abs(E_y).^2))/I0;
S1 = sum(sum(abs(E_x).^2 - abs(E_y).^2))/I0;
S2 = sum(sum(2*real(E_x.*conj(E_y))))/I0;
S3 = sum(sum(2*imag(E_x.*conj(E_y))))/I0;
P = sqrt(S1^2 + S2^2 + S3^2)/S0;
end