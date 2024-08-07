
numBoundaryConditionPoints = [25 25]; 

x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,1,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,1,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
%% 


numInitialConditionPoints  = 50;

x0IC = linspace(-1,1,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);
u0IC = -sin(pi*x0IC);
%% 

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
%% 


numInternalCollocationPoints = 10000;

pointSet = sobolset(2);
points = net(pointSet,numInternalCollocationPoints);

dataX = 2*points(:,1)-1;
dataT = points(:,2);
%% 
% Create an array datastore containing the training data.

ds = arrayDatastore([dataX dataT]);
%% Define Deep Learning Model

% % Prof. Luke asking for simultaneous plot of error due to validation in
% the same graph- ROHAN& ARKA.

numLayers = 9;
numNeurons = 20;
%% 

parameters = struct;

sz = [numNeurons 2];
parameters.fc1.Weights = initializeHe(sz,2);
parameters.fc1.Bias = nullinitialization_Rudra([numNeurons 1]);
%% 
% 

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = nullinitialization_Rudra([numNeurons 1]);
end
%% 

sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = nullinitialize_Arka(sz,numIn);
parameters.("fc" + numLayers).Bias = nullinitialization_Rudra([1 1]);
%% 
%% 

%%

% parameters.fc1

%%


numEpochs = 100;
miniBatchSize = 1000;
%% 

executionEnvironment = "auto";
%% 
initialLearnRate = 0.01;
decayRate = 0.005;
%% Train Network

mbq = minibatchqueue(ds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFormat','BC', ...
    'OutputEnvironment',executionEnvironment);
%% 
% Convert the initial and boundary conditions to |dlarray|. For the input data 
% points, specify format with dimensions |'CB'| (channel, batch).

dlX0 = dlarray(X0,'CB');
dlT0 = dlarray(T0,'CB');
dlU0 = dlarray(U0);
%% 

if (executionEnvironment == "auto" && canUseGPU) || (executionEnvironment == "gpu")
    dlX0 = gpuArray(dlX0);
    dlT0 = gpuArray(dlT0);
    dlU0 = gpuArray(dlU0);
end
%% 

averageGrad = [];
averageSqGrad = [];
%% 
accfun = dlaccelerate(@modelGradients);
%% 
% Initialize the training progress plot.

figure
C = colororder;
lineLoss = animatedline('Color',C(2,:));
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on
%% 

start = tic;

iteration = 0;

for epoch = 1:numEpochs
    
    reset(mbq);

    while hasdata(mbq)
        iteration = iteration + 1;

        dlXT = next(mbq);
        dlX = dlXT(1,:);
        dlT = dlXT(2,:);

        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss] = dlfeval(accfun,parameters,dlX,dlT,dlX0,dlT0,dlU0);

        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end

    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    addpoints(lineLoss,iteration, loss);

    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + loss)
    drawnow
end
%% 
% Check the effectiveness of the accelerated function by checking the hit and 
% occupancy rate.

% accfun
parameters;

%%
%% Evaluate Model Accuracy
 
% Set the target times to test the model at. For each time, calculate the solution 
% at 1001 equally spaced points in the range [-1,1].

tTest = [0.25 0.5 0.75 1];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);

figure

for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);

    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = model(parameters,dlXTest,dlTTest);

    % Calcualte true values.
    UTest = solveBurgers(XTest,t,0.01/pi);

    % solveCancer

    % Calculate error.
    err = norm(extractdata(dlUPred) - UTest) / norm(UTest);

    % Plot predictions.
    subplot(2,2,i)
    plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
    ylim([-1.1, 1.1])

    % Plot true values.
    hold on
    plot(XTest, UTest, '--','LineWidth',2)
    hold off

    title("t = " + t + ", Error = " + gather(err));
end

subplot(2,2,2)
legend('Predicted','True')
%% 
%% Solve Burger's Equation Function - %% Change the equation for Stochastic Centrosome movement
% The |solveBurgers| function returns the true solution of Burger's equation 
% at times |t| as outlined in [2].  CHANGE THIS EQUATION AS PER YOUR MODEL
% FORMAT. I AM ADDING NEW SECTIONS FOR Navier-Stokes and Cahn-Hillard Problem

function U = solveBurgers(X,t,nu)

% Define functions.
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));

% Initialize solutions.
U = zeros(size(X));

% Loop over x values.
for i = 1:numel(X)
    x = X(i);

    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
    if abs(x) ~= 1
        fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
        uxt = -integral(fun,-inf,inf);
        fun = @(eta) f(x-eta) .* g(eta);
        U(i) = uxt / integral(fun,-inf,inf);
    end
end

end

%%

function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlX0,dlT0,dlU0)

% Make predictions with the initial conditions.
U = model(parameters,dlX,dlT);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);

% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);

% Calculate lossU. Enforce initial and boundary conditions.
dlU0Pred = model(parameters,dlX0,dlT0);
lossU = mse(dlU0Pred, dlU0);

% Combine losses.
loss = lossF + lossU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
%% Model Function
% The model trained in this example consists of a series of fully connect operations 
% with a tanh operation between each one.
% 
% The model function takes as input the model parameters |parameters| and the 
% network inputs |dlX| and |dlT|, and returns the model output |dlU|.

function dlU = model(parameters,dlX,dlT)

dlXT = [dlX;dlT];
numLayers = numel(fieldnames(parameters));

% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
dlU = fullyconnect(dlXT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;

    dlU = tanh(dlU);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;

    dlU = fullyconnect(dlU, weights, bias);
end

end
%% 