function plot_follower_log(csv_path)
% =========================================================
% PLOT - Trayectorias Lider y Seguidor desde log de Python
% Lee el CSV generado por MavlinkControlLFLog.py y genera
% las mismas gráficas que el script original de Simulink
% =========================================================

if nargin < 1
    % Buscar el archivo más reciente si no se especifica
    files = dir('lf_log_*.csv');
    if isempty(files)
        error('No se encontraron archivos lf_log_*.csv en el directorio actual.');
    end
    [~, idx] = max([files.datenum]);
    csv_path = files(idx).name;
    fprintf('Usando archivo: %s\n', csv_path);
end

% --- Leer CSV ---
fprintf('Leyendo archivo %s...\n', csv_path);
data = readtable(csv_path);

% --- Extraer variables ---
t      = data.time;
% Lider real (del CSV del círculo)
xL     = data.lx;
yL     = data.ly;
zL     = data.lz;
psiL   = data.l_yaw;
% Seguidor real
xS     = data.sx;
yS     = data.sy;
zS     = data.sz;
psiS   = data.s_yaw;
% Setpoints del lider (del CSV del círculo)
xdesL  = data.lx_sp;
ydesL  = data.ly_sp;
zdesL  = data.lz_sp;
% Setpoints del seguidor
xdesS  = data.xd;
ydesS  = data.yd;
zdesS  = data.zd;
psidesS = psiL;  % El seguidor sigue el yaw del lider

% Parámetros de formación (fijos del código Python)
% Se estiman a partir de los datos
d  = mean(sqrt((xL-xS).^2 + (yL-yS).^2));  % distancia media real
dz = mean(zdesS - zL);  % diferencia de altitud deseada

% --- Colores ---
C_refL = [0.40 0.65 1.00];
C_refS = [1.00 0.60 0.40];
C_L    = [0.10 0.35 0.75];
C_S    = [0.80 0.15 0.10];
lw  = 2.0;
lws = 1.2;
sc_drone = 0.50;
fw_arrow = 2.8;

% =========================================================
% FIGURA 1 - Trayectoria XY con frames y vector d
% =========================================================
figure('Name','Trayectoria XY','Color','w','Position',[50 50 800 800])
hold on

% Setpoints
plot(xdesL, ydesL, '--', 'Color', C_refL, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Lider');
plot(xdesS, ydesS, '--', 'Color', C_refS, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Seguidor');

% Trayectorias reales
plot(xL, yL, '-', 'Color', C_L, 'LineWidth', lw, ...
     'DisplayName', 'Lider real');
plot(xS, yS, '-', 'Color', C_S, 'LineWidth', lw, ...
     'DisplayName', 'Seguidor real');

% Puntos de inicio
plot(xL(1), yL(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_L, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio L');
plot(xS(1), yS(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_S, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio S');

% Vectores d (muchos, distribuidos uniformemente)
n_d = 24;
idx_d = round(linspace(1, length(t), n_d+2));
idx_d = idx_d(2:end-1);

for k = idx_d
    if k > length(xL) || k > length(xS), continue; end
    quiver(xL(k), yL(k), xS(k)-xL(k), yS(k)-yL(k), 0, ...
        'Color', [0.50 0.50 0.50], 'LineWidth', 1.8, ...
        'MaxHeadSize', 0.30, 'HandleVisibility', 'off');
end

% Frames de los drones (pocos instantes, primer punto = inicio)
n_frames = 5;
idx_rest = round(linspace(2, length(t), n_frames+1));
idx_rest = idx_rest(1:end-1);
idx = [1, idx_rest];

for ki = 1:length(idx)
    k = idx(ki);
    if k > length(xL) || k > length(xS), continue; end

    xLk = xL(k); yLk = yL(k); psiLk = psiL(k);
    xSk = xS(k); ySk = yS(k); psiSk = psiS(k);

    % Ejes del lider
    quiver(xLk, yLk, sc_drone*cos(psiLk),      sc_drone*sin(psiLk),      0, ...
        'Color', [0.85 0.10 0.10], 'LineWidth', fw_arrow, 'MaxHeadSize', 0.55, ...
        'HandleVisibility', 'off');
    quiver(xLk, yLk, sc_drone*cos(psiLk+pi/2), sc_drone*sin(psiLk+pi/2), 0, ...
        'Color', [0.10 0.65 0.10], 'LineWidth', fw_arrow, 'MaxHeadSize', 0.55, ...
        'HandleVisibility', 'off');

    % Ejes del seguidor
    quiver(xSk, ySk, sc_drone*cos(psiSk),      sc_drone*sin(psiSk),      0, ...
        'Color', [0.85 0.10 0.10], 'LineWidth', fw_arrow, 'MaxHeadSize', 0.55, ...
        'HandleVisibility', 'off');
    quiver(xSk, ySk, sc_drone*cos(psiSk+pi/2), sc_drone*sin(psiSk+pi/2), 0, ...
        'Color', [0.10 0.65 0.10], 'LineWidth', fw_arrow, 'MaxHeadSize', 0.55, ...
        'HandleVisibility', 'off');
end

% Frame RTK en el origen
sc_rtk  = sc_drone * 2.0;
fw_rtk  = fw_arrow + 0.8;
quiver(0, 0, sc_rtk, 0, 0, ...
    'Color', [0.85 0.10 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
quiver(0, 0, 0, sc_rtk, 0, ...
    'Color', [0.10 0.65 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
text(sc_rtk+0.15, 0.05, '$\hat{x}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', 'HandleVisibility', 'off');
text(-0.05, sc_rtk+0.15, '$\hat{y}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', ...
    'HorizontalAlignment', 'center', 'HandleVisibility', 'off');
plot(0, 0, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');

% Entradas de leyenda para los ejes y el vector d 
plot(nan, nan, '-', 'Color', [0.85 0.10 0.10], 'LineWidth', 2.5, ...
    'DisplayName', '$\hat{x}_D$');
plot(nan, nan, '-', 'Color', [0.10 0.65 0.10], 'LineWidth', 2.5, ...
    'DisplayName', '$\hat{y}_D$');
plot(nan, nan, '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 2.0, ...
    'DisplayName', 'Vector $\mathbf{d}$');

axis equal; grid on; box on
xlabel('$x_{RTK}$ [m]', 'Interpreter', 'latex', 'FontSize', 11)
ylabel('$y_{RTK}$ [m]', 'Interpreter', 'latex', 'FontSize', 11)
title('Trayectoria XY con marcos de referencia y vector de formacion')
legend('Location', 'best', 'Interpreter', 'latex')

% =========================================================
% FIGURA 2 - Lider: setpoint vs real
% =========================================================
figure('Name','Lider: Setpoint vs Real','Color','w','Position',[80 80 700 700])

plot(xdesL, ydesL, '--', 'Color', C_refL, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Lider'); hold on
plot(xL, yL, '-', 'Color', C_L, 'LineWidth', lw, ...
     'DisplayName', 'Lider real')
plot(xL(1), yL(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_L, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio')

% Frame RTK en el origen
quiver(0, 0, sc_rtk, 0, 0, ...
    'Color', [0.85 0.10 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
quiver(0, 0, 0, sc_rtk, 0, ...
    'Color', [0.10 0.65 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
text(sc_rtk+0.15, 0.05, '$\hat{x}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', 'HandleVisibility', 'off');
text(-0.05, sc_rtk+0.15, '$\hat{y}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', ...
    'HorizontalAlignment', 'center', 'HandleVisibility', 'off');
plot(0, 0, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');

axis equal; grid on; box on
xlabel('x [m]'); ylabel('y [m]')
title('Lider: Setpoint vs Trayectoria Real')
legend('Location','best')

% =========================================================
% FIGURA 3 - Seguidor: setpoint vs real
% =========================================================
figure('Name','Seguidor: Setpoint vs Real','Color','w','Position',[110 110 700 700])

plot(xdesS, ydesS, '--', 'Color', C_refS, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Seguidor'); hold on
plot(xS, yS, '-', 'Color', C_S, 'LineWidth', lw, ...
     'DisplayName', 'Seguidor real')
plot(xS(1), yS(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_S, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio')

% Frame RTK en el origen
quiver(0, 0, sc_rtk, 0, 0, ...
    'Color', [0.85 0.10 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
quiver(0, 0, 0, sc_rtk, 0, ...
    'Color', [0.10 0.65 0.10], 'LineWidth', fw_rtk, 'MaxHeadSize', 0.45, ...
    'HandleVisibility', 'off');
text(sc_rtk+0.15, 0.05, '$\hat{x}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', 'HandleVisibility', 'off');
text(-0.05, sc_rtk+0.15, '$\hat{y}_{RTK}$', 'FontSize', 12, ...
    'FontWeight', 'bold', 'Interpreter', 'latex', ...
    'HorizontalAlignment', 'center', 'HandleVisibility', 'off');
plot(0, 0, 'ok', 'MarkerSize', 8, 'MarkerFaceColor', 'k', ...
    'LineWidth', 1.5, 'HandleVisibility', 'off');

axis equal; grid on; box on
xlabel('x [m]'); ylabel('y [m]')
title('Seguidor: Setpoint vs Trayectoria Real')
legend('Location','best')

% =========================================================
% FIGURA 4 - Posicion x, y, z, psi vs tiempo
% =========================================================
figure('Name','Estados vs Tiempo','Color','w','Position',[140 140 950 750])

subplot(4,1,1)
plot(t, xdesL, '--', 'Color', C_refL, 'LineWidth', lws, 'DisplayName', 'Setpoint L'); hold on
plot(t, xdesS, '--', 'Color', C_refS, 'LineWidth', lws, 'DisplayName', 'Setpoint S')
plot(t, xL,    '-',  'Color', C_L,    'LineWidth', lw,  'DisplayName', 'x Lider')
plot(t, xS,    '-',  'Color', C_S,    'LineWidth', lw,  'DisplayName', 'x Seguidor')
ylabel('x [m]'); grid on; legend('Location','best','NumColumns',2)
title('Posicion X vs Tiempo')

subplot(4,1,2)
plot(t, ydesL, '--', 'Color', C_refL, 'LineWidth', lws, 'DisplayName', 'Setpoint L'); hold on
plot(t, ydesS, '--', 'Color', C_refS, 'LineWidth', lws, 'DisplayName', 'Setpoint S')
plot(t, yL,    '-',  'Color', C_L,    'LineWidth', lw,  'DisplayName', 'y Lider')
plot(t, yS,    '-',  'Color', C_S,    'LineWidth', lw,  'DisplayName', 'y Seguidor')
ylabel('y [m]'); grid on; legend('Location','best','NumColumns',2)
title('Posicion Y vs Tiempo')

subplot(4,1,3)
plot(t, zdesS, '--', 'Color', C_refS, 'LineWidth', lws, 'DisplayName', 'Setpoint S'); hold on
plot(t, zL,    '-',  'Color', C_L,    'LineWidth', lw,  'DisplayName', 'z Lider')
plot(t, zS,    '-',  'Color', C_S,    'LineWidth', lw,  'DisplayName', 'z Seguidor')
ylabel('z [m]'); grid on; legend('Location','best','NumColumns',2)
title('Posicion Z (Altitud) vs Tiempo')

subplot(4,1,4)
plot(t, rad2deg(psidesS), '--', 'Color', C_refS, 'LineWidth', lws, 'DisplayName', 'Setpoint S'); hold on
plot(t, rad2deg(psiL),    '-',  'Color', C_L,    'LineWidth', lw,  'DisplayName', 'psi Lider')
plot(t, rad2deg(psiS),    '-',  'Color', C_S,    'LineWidth', lw,  'DisplayName', 'psi Seguidor')
ylabel('psi [deg]'); xlabel('Tiempo [s]'); grid on
legend('Location','best','NumColumns',2)
title('Yaw vs Tiempo')

% =========================================================
% FIGURA 5 - Errores del lider (usando errores del CSV)
% =========================================================
% Verificar si existen columnas de error del lider
if ismember('lex', data.Properties.VariableNames) && ismember('ley', data.Properties.VariableNames)
    ex_L = data.lex;
    ey_L = data.ley;
else
    % Calcular errores si no existen
    ex_L = xdesL - xL;
    ey_L = ydesL - yL;
end
ez_L   = zeros(size(t));  % Asumimos que z_setpoint = z_real para el lider
epsi_L = rad2deg(atan2(sin(psidesS - psiL), cos(psidesS - psiL)));

figure('Name','Errores Lider','Color','w','Position',[170 170 950 700])

subplot(4,1,1)
plot(t, ex_L, '-', 'Color', C_L, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_x [m]'); title('Errores del Lider respecto a su setpoint')
legend('e_x Lider')

subplot(4,1,2)
plot(t, ey_L, '-', 'Color', C_L, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_y [m]'); legend('e_y Lider')

subplot(4,1,3)
plot(t, ez_L, '-', 'Color', C_L, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_z [m]'); legend('e_z Lider')

subplot(4,1,4)
plot(t, epsi_L, '-', 'Color', C_L, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_psi [deg]'); xlabel('Tiempo [s]'); legend('e_psi Lider')

% =========================================================
% FIGURA 6 - Errores del seguidor
% =========================================================
if ismember('ex', data.Properties.VariableNames)
    ex_S = data.ex;
    ey_S = data.ey;
    ez_S = data.ez;
else
    % Calcular errores si no existen
    ex_S = xdesS - xS;
    ey_S = ydesS - yS;
    ez_S = zdesS - zS;
end
epsi_S = rad2deg(atan2(sin(psidesS - psiS), cos(psidesS - psiS)));

figure('Name','Errores Seguidor','Color','w','Position',[200 200 950 700])

subplot(4,1,1)
plot(t, ex_S, '-', 'Color', C_S, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_x [m]'); title('Errores del Seguidor respecto a su setpoint')
legend('e_x Seguidor')

subplot(4,1,2)
plot(t, ey_S, '-', 'Color', C_S, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_y [m]'); legend('e_y Seguidor')

subplot(4,1,3)
plot(t, ez_S, '-', 'Color', C_S, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_z [m]'); legend('e_z Seguidor')

subplot(4,1,4)
plot(t, epsi_S, '-', 'Color', C_S, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_psi [deg]'); xlabel('Tiempo [s]'); legend('e_psi Seguidor')

% =========================================================
% FIGURA 7 - Distancia real entre lider y seguidor
% =========================================================
dist = sqrt((xL-xS).^2 + (yL-yS).^2 + (zL-zS).^2);
dist_horiz = sqrt((xL-xS).^2 + (yL-yS).^2);

figure('Name','Distancia Formacion','Color','w','Position',[230 230 950 380])
plot(t, dist_horiz, '-', 'Color', [0.25 0.75 0.45], 'LineWidth', lw, ...
     'DisplayName', 'Distancia real L-S'); hold on
yline(d, '--', 'Color', [0.0 0.5 0.2], 'LineWidth', 1.2, ...
      'Label', sprintf('d = %.1f m', d), 'DisplayName', 'Distancia deseada')
xlabel('Tiempo [s]'); ylabel('Distancia [m]')
title('Distancia entre Lider y Seguidor vs Tiempo')
grid on; box on; legend('Location','best')

% =========================================================
% METRICAS
% =========================================================
fprintf('\n============= METRICAS =============\n')
fprintf('             Lider      Seguidor\n')
fprintf('RMS error x: %6.4f m   %6.4f m\n',  rms(ex_L), rms(ex_S))
fprintf('RMS error y: %6.4f m   %6.4f m\n',  rms(ey_L), rms(ey_S))
fprintf('RMS error z: %6.4f m   %6.4f m\n',  rms(ez_L), rms(ez_S))
fprintf('RMS error psi:%5.4f deg  %5.4f deg\n', rms(epsi_L), rms(epsi_S))
fprintf('------------------------------------\n')
fprintf('Distancia deseada L-S:  %.4f m\n', d)
fprintf('Distancia media  L-S:   %.4f m\n', mean(dist_horiz))
fprintf('Distancia max    L-S:   %.4f m\n', max(dist_horiz))
fprintf('Distancia min    L-S:   %.4f m\n', min(dist_horiz))
fprintf('Error formacion RMS:    %.4f m\n', rms(dist_horiz - d))
fprintf('====================================\n\n')

end