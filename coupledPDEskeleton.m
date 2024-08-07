% Define parameters
x0 = 3;
Delt = 1;
Tmax = 50;

kona = 16;
atot = 0.2;
rap = 0.01;
konp = 1;
rpa = 2;
konm = 0.01;
kam = 1;
kpm = 0.01;
koffm = 1;
Ka = 0.1;
Kp = 0.1;

d1 = 0.02;
d2 = 0.02;
d3 = 0.02;
mu = 0.02;

% Define spatial domain
x = linspace(0, 6, 100);
dx = x(2) - x(1);

% Initial values
a = 0.565 * 0.5 * (1 - tanh((x - x0) / 0.1));
p = 0.3 * (1 + tanh((x - x0) / 0.1));
m = 0.582 * 0.5 * (1 - tanh((x - x0) / 0.1));

% Time vector
t = 0:Delt:Tmax;

% Preallocate results
a_results = zeros(length(t), length(x));
p_results = zeros(length(t), length(x));
m_results = zeros(length(t), length(x));

a_results(1, :) = a;
p_results(1, :) = p;
m_results(1, :) = m;

% Simulation loop
for n = 2:length(t)
    % Compute gradients
    grad_a = gradient(a, dx);
    grad_p = gradient(p, dx);
    grad_m = gradient(m, dx);
    
    div_grad_a = gradient(grad_a, dx);
    div_grad_p = gradient(grad_p, dx);
    div_grad_m = gradient(grad_m, dx);
    
    div_a_grad_m = gradient(a .* grad_m, dx);
    div_p_grad_m = gradient(p .* grad_m, dx);
    div_m_grad_m = gradient(m .* grad_m, dx);
    
    % Update equations
    dt_a = (kona * (1 - a) * (atot - a)^2 - a - rap * a * p + d1 * div_grad_a - mu * div_a_grad_m);
    dt_p = (konp - konp * p - rpa * p * a + d2 * div_grad_p - mu * div_p_grad_m);
    dt_m = (konm + kam * a / (Ka + a) + kpm / (Kp + p) - koffm * m + d3 * div_grad_m - mu * div_m_grad_m);
    
    % Forward Euler method
    a = a + Delt * dt_a;
    p = p + Delt * dt_p;
    m = m + Delt * dt_m;
    
    % Store results
    a_results(n, :) = a;
    p_results(n, :) = p;
    m_results(n, :) = m;
end

% Plot results
figure;
subplot(3, 1, 1);
surf(x, t, a_results);
title('a');
xlabel('x');
ylabel('Time');
zlabel('a');
shading interp;

subplot(3, 1, 2);
surf(x, t, p_results);
title('p');
xlabel('x');
ylabel('Time');
zlabel('p');
shading interp;

subplot(3, 1, 3);
surf(x, t, m_results);
title('m');
xlabel('x');
ylabel('Time');
zlabel('m');
shading interp;