% Par Protein Model in Ellipsoidal Geometry
clear all

% Define parameters
b1 = 0.011979;
b2 = 4.6526;
b3 = 4.177344;
b4 = 0.911258;
b5 = 0.011441;
b6 = 5.003463;
b7 = 0.073918;
b8 = 0.082689;
b9 = 0.724544;
b10 = 0.1;
b11 = 0.1;
b12 = 0.1;
mu = 0.002;
ay = 1;
ry = 1;
d1 = 0.002;
d2 = 0.0015;
d3 = 0.002;

k = 1;
x0 = 2.5;
Delt = 100;
Tmax = 10000;

R1 = 3;
R2 = 1.3;
th1 = 0.3;
th2 = 0.3;

phi = 90;
psi = 60;

% Define spatial domain
nx = 100;
ny = 100;
x = linspace(-R1, R1, nx);
y = linspace(-R2, R2, ny);
[xx, yy] = meshgrid(x, y);
dx = x(2) - x(1);
dy = y(2) - y(1);

% Initial values
a1 = 0.565 * 0.5 * (1 - tanh((xx - x0) / 0.1));
a10 = 0.316 * 0.5 * (1 - tanh((xx - x0) / 0.1));
a11 = 0.582 * 0.5 * (1 - tanh((xx - x0) / 0.1));
p = 0.3 * (1 + tanh((xx - x0) / 0.1));
m = 0.6973 * 0.5 * (1 - tanh((xx - x0) / 0.1));

% Time vector
t = 0:Delt:Tmax;

% Preallocate results
a1_results = zeros(nx, ny, length(t));
a10_results = zeros(nx, ny, length(t));
a11_results = zeros(nx, ny, length(t));
p_results = zeros(nx, ny, length(t));
m_results = zeros(nx, ny, length(t));

a1_results(:, :, 1) = a1;
a10_results(:, :, 1) = a10;
a11_results(:, :, 1) = a11;
p_results(:, :, 1) = p;
m_results(:, :, 1) = m;

% Simulation loop
for n = 2:length(t)
    % Compute gradients and divergences
    [grad_a1x, grad_a1y] = gradient(a1, dx, dy);
    [grad_a10x, grad_a10y] = gradient(a10, dx, dy);
    [grad_a11x, grad_a11y] = gradient(a11, dx, dy);
    [grad_px, grad_py] = gradient(p, dx, dy);
    [grad_mx, grad_my] = gradient(m, dx, dy);
    
    div_grad_a1 = divergence(grad_a1x, grad_a1y);
    div_grad_a10 = divergence(grad_a10x, grad_a10y);
    div_grad_a11 = divergence(grad_a11x, grad_a11y);
    div_grad_p = divergence(grad_px, grad_py);
    div_grad_m = divergence(grad_mx, grad_my);
    
    div_a1_grad_m = divergence(a1 .* grad_mx, a1 .* grad_my);
    div_a10_grad_m = divergence(a10 .* grad_mx, a10 .* grad_my);
    div_a11_grad_m = divergence(a11 .* grad_mx, a11 .* grad_my);
    div_p_grad_m = divergence(p .* grad_mx, p .* grad_my);
    div_m_grad_m = divergence(m .* grad_mx, m .* grad_my);
    
    % Update equations
    dt_a1 = (b1 * ay - a1 - b2 * ay * a1 + b3 * a10 - 2 * b2 * a1.^2 + 2 * b3 * a11 - b4 * p .* a1 + d1 * div_grad_a1 - mu * div_a1_grad_m);
    dt_a10 = (b5 * ay^2 - a10 + b2 * ay * a1 - b3 * a10 - b6 * a10 + a11 - b4 * p .* a10 + d1 * div_grad_a10 - mu * div_a10_grad_m);
    dt_a11 = (b2 * a1.^2 - b3 * a11 - a11 + b6 * a10 - 2 * b4 * p .* a11 + d1 * div_grad_a11 - mu * div_a11_grad_m);
    dt_p = (ry * b7 - b8 * p - b9 * (a1 + a10 + 2 * a11) .* p + d2 * div_grad_p - mu * div_p_grad_m);
    dt_m = (b10 * (b11 / (b11 + p)) - b12 * m + d3 * div_grad_m - mu * div_m_grad_m);
    
    % Forward Euler method
    a1 = a1 + Delt * dt_a1;
    a10 = a10 + Delt * dt_a10;
    a11 = a11 + Delt * dt_a11;
    p = p + Delt * dt_p;
    m = m + Delt * dt_m;
    
    % Store results
    a1_results(:, :, n) = a1;
    a10_results(:, :, n) = a10;
    a11_results(:, :, n) = a11;
    p_results(:, :, n) = p;
    m_results(:, :, n) = m;
end

% Plot results
figure;
subplot(3, 2, 1);
surf(x, y, a1_results(:, :, end));
title('a1');
xlabel('x');
ylabel('y');
zlabel('a1');
shading interp;

subplot(3, 2, 2);
surf(x, y, a10_results(:, :, end));
title('a10');
xlabel('x');
ylabel('y');
zlabel('a10');
shading interp;

subplot(3, 2, 3);
surf(x, y, a11_results(:, :, end));
title('a11');
xlabel('x');
ylabel('y');
zlabel('a11');
shading interp;

subplot(3, 2, 4);
surf(x, y, p_results(:, :, end));
title('p');
xlabel('x');
ylabel('y');
zlabel('p');
shading interp;

subplot(3, 2, 5);
surf(x, y, m_results(:, :, end));
title('m');
xlabel('x');
ylabel('y');
zlabel('m');
shading interp;