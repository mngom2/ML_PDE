t = 3*pi; % total simulation time
T = pi; % period
x = -t:2*t/100:t;
% lets say your original function is y=2*x which repeated every T second, then
y = mod(x.^2+pi,2*pi)-pi;
plot(x,y)