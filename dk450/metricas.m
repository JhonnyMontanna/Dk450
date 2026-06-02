% =========================================================
%  PLOT - Trayectorias Lider y Seguidor
%  Extrae variables del objeto out de Simulink
%  CON MÉTRICAS INTEGRALES (IAE, ISE, ITAE)
% =========================================================

if ~exist('out','var')
    error('No existe el objeto "out". Simula el modelo primero.');
end

% --- Parámetros de formación ---
d = 1.0;      % Distancia deseada L-S [m]
a = -pi/2;    % Ángulo del offset [rad]
dz = 1.0;     % Diferencia de altitud [m]
h = 4.0;      % Altitud de referencia del líder [m]

% --- Extraer del objeto out ---
xL     = out.xL.Data(:);
yL     = out.yL.Data(:);
zL     = out.zL.Data(:);
psiL   = out.psiL.Data(:);
xS     = out.xS.Data(:);
yS     = out.yS.Data(:);
zS     = out.zS.Data(:);
psiS   = out.psiS.Data(:);
xdesL  = out.xdes.Data(:);
ydesL  = out.ydes.Data(:);
t      = out.tout(:);

% Setpoint del seguidor
xdesS   = xL + d * cos(psiL + a);
ydesS   = yL + d * sin(psiL + a);
zdesS   = zL + dz;
psidesS = psiL;

% --- Colores ---
C_refL = [0.40 0.65 1.00];
C_refS = [1.00 0.60 0.40];
C_L    = [0.10 0.35 0.75];
C_S    = [0.80 0.15 0.10];
lw  = 2.0;
lws = 1.2;

% =========================================================
%  CÁLCULO DE ERRORES
% =========================================================

% Errores del líder
ex_L   = xdesL - xL;
ey_L   = ydesL - yL;
ez_L   = h - zL;
epsi_L = wrapToPi(psidesS - psiL);

% Errores del seguidor
ex_S   = xdesS - xS;
ey_S   = ydesS - yS;
ez_S   = zdesS - zS;
epsi_S = wrapToPi(psidesS - psiS);

% Error de formación (distancia L-S)
dist = sqrt((xL - xS).^2 + (yL - yS).^2);
err_form = dist - d;

% Error horizontal del seguidor
err_xy_S = sqrt(ex_S.^2 + ey_S.^2);

% =========================================================
%  MÉTRICAS INTEGRALES (IAE, ISE, ITAE)
% =========================================================

fprintf('\n=============================================================')
fprintf('\n  MÉTRICAS INTEGRALES DE ERROR (IAE, ISE, ITAE)')
fprintf('\n=============================================================')

% Función para calcular IAE, ISE, ITAE
compute_metrics = @(error, dt) struct(...
    'IAE',  trapz(abs(error)) * dt, ...
    'ISE',  trapz(error.^2) * dt, ...
    'ITAE', trapz(t .* abs(error)) * dt);

dt = t(2) - t(1);  % Asumiendo paso de tiempo constante

% Calcular métricas para cada error
metrics_L.x   = compute_metrics(ex_L, dt);
metrics_L.y   = compute_metrics(ey_L, dt);
metrics_L.z   = compute_metrics(ez_L, dt);
metrics_L.psi = compute_metrics(epsi_L, dt);

metrics_S.x   = compute_metrics(ex_S, dt);
metrics_S.y   = compute_metrics(ey_S, dt);
metrics_S.z   = compute_metrics(ez_S, dt);
metrics_S.psi = compute_metrics(epsi_S, dt);
metrics_S.xy  = compute_metrics(err_xy_S, dt);

metrics_form = compute_metrics(err_form, dt);

fprintf('\n----------------------------------------')
fprintf('\n  LÍDER (respecto a su setpoint)')
fprintf('\n----------------------------------------')
fprintf('\n  Variable     IAE [m·s]    ISE [m²·s]   ITAE [m·s²]')
fprintf('\n  ------------------------------------------------')
fprintf('\n  e_x          %10.4f   %10.4f   %10.4f', metrics_L.x.IAE, metrics_L.x.ISE, metrics_L.x.ITAE)
fprintf('\n  e_y          %10.4f   %10.4f   %10.4f', metrics_L.y.IAE, metrics_L.y.ISE, metrics_L.y.ITAE)
fprintf('\n  e_z          %10.4f   %10.4f   %10.4f', metrics_L.z.IAE, metrics_L.z.ISE, metrics_L.z.ITAE)
fprintf('\n  e_ψ [rad]    %10.4f   %10.4f   %10.4f', metrics_L.psi.IAE, metrics_L.psi.ISE, metrics_L.psi.ITAE)

fprintf('\n\n----------------------------------------')
fprintf('\n  SEGUIDOR (respecto a su setpoint)')
fprintf('\n----------------------------------------')
fprintf('\n  Variable     IAE [m·s]    ISE [m²·s]   ITAE [m·s²]')
fprintf('\n  ------------------------------------------------')
fprintf('\n  e_x          %10.4f   %10.4f   %10.4f', metrics_S.x.IAE, metrics_S.x.ISE, metrics_S.x.ITAE)
fprintf('\n  e_y          %10.4f   %10.4f   %10.4f', metrics_S.y.IAE, metrics_S.y.ISE, metrics_S.y.ITAE)
fprintf('\n  e_z          %10.4f   %10.4f   %10.4f', metrics_S.z.IAE, metrics_S.z.ISE, metrics_S.z.ITAE)
fprintf('\n  e_ψ [rad]    %10.4f   %10.4f   %10.4f', metrics_S.psi.IAE, metrics_S.psi.ISE, metrics_S.psi.ITAE)
fprintf('\n  error_xy     %10.4f   %10.4f   %10.4f', metrics_S.xy.IAE, metrics_S.xy.ISE, metrics_S.xy.ITAE)

fprintf('\n\n----------------------------------------')
fprintf('\n  FORMACIÓN')
fprintf('\n----------------------------------------')
fprintf('\n  Variable          IAE [m·s]    ISE [m²·s]   ITAE [m·s²]')
fprintf('\n  ----------------------------------------------------')
fprintf('\n  error_formación   %10.4f   %10.4f   %10.4f', metrics_form.IAE, metrics_form.ISE, metrics_form.ITAE)

fprintf('\n\n=============================================================\n')

% =========================================================
%  FIGURA 1 - Trayectoria XY con frames y vector d
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

%  Vectores d (muchos, distribuidos uniformemente)
n_d = 24;
idx_d = round(linspace(1, length(t), n_d+2));
idx_d = idx_d(2:end-1);

for k = idx_d
    if k > length(xL) || k > length(xS), continue; end
    quiver(xL(k), yL(k), xS(k)-xL(k), yS(k)-yL(k), 0, ...
        'Color', [0.50 0.50 0.50], 'LineWidth', 1.8, ...
        'MaxHeadSize', 0.30, 'HandleVisibility', 'off');
end

%  Frames de los drones (pocos instantes, primer punto = inicio)
n_frames = 5;
idx_rest = round(linspace(2, length(t), n_frames+1));
idx_rest = idx_rest(1:end-1);
idx = [1, idx_rest];

sc_drone = 0.50;
fw_arrow = 2.8;

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


%  Frame RTK en el origen (mismo estilo que los drones, mas grande)
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

%  Entradas de leyenda para los ejes y el vector d 
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
%  FIGURA 2 - Lider: setpoint vs real
% =========================================================
figure('Name','Lider: Setpoint vs Real','Color','w','Position',[80 80 700 700])

plot(xdesL, ydesL, '--', 'Color', C_refL, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Lider'); hold on
plot(xL, yL, '-', 'Color', C_L, 'LineWidth', lw, ...
     'DisplayName', 'Lider real')
plot(xL(1), yL(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_L, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio')

%  Frame RTK en el origen
sc_rtk = sc_drone * 2.0;
fw_rtk = fw_arrow + 0.8;
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
%  FIGURA 3 - Seguidor: setpoint vs real
% =========================================================
figure('Name','Seguidor: Setpoint vs Real','Color','w','Position',[110 110 700 700])

plot(xdesS, ydesS, '--', 'Color', C_refS, 'LineWidth', lws, ...
     'DisplayName', 'Setpoint Seguidor'); hold on
plot(xS, yS, '-', 'Color', C_S, 'LineWidth', lw, ...
     'DisplayName', 'Seguidor real')
plot(xS(1), yS(1), 'o', 'MarkerSize', 10, ...
     'MarkerFaceColor', C_S, 'MarkerEdgeColor', 'k', ...
     'DisplayName', 'Inicio')

%  Frame RTK en el origen
sc_rtk = sc_drone * 2.0;
fw_rtk = fw_arrow + 0.8;
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
%  FIGURA 4 - Posicion x, y, z, psi vs tiempo
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
%  FIGURA 5 - Errores del lider
% =========================================================
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
plot(t, rad2deg(epsi_L), '-', 'Color', C_L, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_psi [deg]'); xlabel('Tiempo [s]'); legend('e_psi Lider')

% =========================================================
%  FIGURA 6 - Errores del seguidor
% =========================================================
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
plot(t, rad2deg(epsi_S), '-', 'Color', C_S, 'LineWidth', lw)
yline(0,'k--','LineWidth',0.8); grid on
ylabel('e_psi [deg]'); xlabel('Tiempo [s]'); legend('e_psi Seguidor')

% =========================================================
%  FIGURA 7 - Distancia real entre lider y seguidor
% =========================================================
figure('Name','Distancia Formacion','Color','w','Position',[230 230 950 380])
plot(t, dist, '-', 'Color', [0.25 0.75 0.45], 'LineWidth', lw, ...
     'DisplayName', 'Distancia real L-S'); hold on
yline(d, '--', 'Color', [0.0 0.5 0.2], 'LineWidth', 1.2, ...
      'Label', sprintf('d = %.1f m', d), 'DisplayName', 'Distancia deseada')
xlabel('Tiempo [s]'); ylabel('Distancia [m]')
title('Distancia entre Lider y Seguidor vs Tiempo')
grid on; box on; legend('Location','best')

% =========================================================
%  FIGURA 8 - Error de formación (distancia - d)
% =========================================================
figure('Name','Error de Formación','Color','w','Position',[260 260 950 380])
plot(t, err_form, '-', 'Color', [0.85 0.55 0.10], 'LineWidth', lw, ...
     'DisplayName', 'Error de formacion (dist - d)'); hold on
yline(0, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Referencia')
yline(0.1, 'r:', 'LineWidth', 1.0, 'DisplayName', '±0.1 m')
yline(-0.1, 'r:', 'LineWidth', 1.0, 'HandleVisibility', 'off')
xlabel('Tiempo [s]'); ylabel('Error [m]')
title('Error de Formación vs Tiempo')
grid on; box on; legend('Location','best')

% =========================================================
%  FIGURA 9 - Error XY del seguidor
% =========================================================
figure('Name','Error XY Seguidor','Color','w','Position',[290 290 950 380])
plot(t, err_xy_S, '-', 'Color', [0.75 0.25 0.65], 'LineWidth', lw, ...
     'DisplayName', 'Error XY del seguidor')
yline(0, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Referencia')
xlabel('Tiempo [s]'); ylabel('Error XY [m]')
title('Error Horizontal del Seguidor vs Tiempo')
grid on; box on; legend('Location','best')

% =========================================================
%  MÉTRICAS RMS (originales)
% =========================================================
fprintf('\n============= MÉTRICAS RMS =============\n')
fprintf('             Lider      Seguidor\n')
fprintf('RMS error x: %6.4f m   %6.4f m\n',  rms(ex_L), rms(ex_S))
fprintf('RMS error y: %6.4f m   %6.4f m\n',  rms(ey_L), rms(ey_S))
fprintf('RMS error z: %6.4f m   %6.4f m\n',  rms(ez_L), rms(ez_S))
fprintf('RMS error psi:%5.4f deg  %5.4f deg\n', rad2deg(rms(epsi_L)), rad2deg(rms(epsi_S)))
fprintf('------------------------------------\n')
fprintf('Distancia deseada L-S:  %.4f m\n', d)
fprintf('Distancia media  L-S:   %.4f m\n', mean(dist))
fprintf('Distancia max    L-S:   %.4f m\n', max(dist))
fprintf('Distancia min    L-S:   %.4f m\n', min(dist))
fprintf('Error formacion RMS:    %.4f m\n', rms(err_form))
fprintf('====================================\n\n')

% Función auxiliar para RMS
function y = rms(x)
    y = sqrt(mean(x.^2));
end

% Función auxiliar para wrapToPi (si no está disponible)
function y = wrapToPi(x)
    y = mod(x + pi, 2*pi) - pi;
end