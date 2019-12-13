fun = @(x) ObjFunfmincon(x);

nonlcon = @NonLinearIneq;
A = [];
b = [];
Aeq = [];
beq = [];


%%%%%Initialization\%%%%

yp0 = 37;     % 37 for 50 et 25yp0 - 13 needs to be a multiple of the wavelength lambda0=8
xp0 = -16;%-64;      %-16;
lnx1 = linspace(0,32,5); %number of dielectrics in the x direction
lny = linspace(0,32,5);  %number of dielectrics in the y direction
[xp,yp] = ndgrid(lnx1,lny);
 
xp = xp + xp0; 

yp = yp + yp0;

x = xp(:);
y = yp(:);

nc = length(x);

theta = -0*pi/9; %%
for ii = 1:nc
    x(ii) = x(ii) * cos(theta) - y(ii) *sin(theta);
    y(ii) = sin(theta) * x(ii) + cos(theta)*y(ii) ;
%     x(ii+nc/2) = x(ii+nc/2) * cos(-theta) - y(ii+nc/2) *sin(-theta);
%     y(ii+nc/2) = sin(-theta) * x(ii+nc/2) + cos(-theta)*y(ii+nc/2) ;
end
xp = x;
yp = y;
X0 =[xp' yp'];
%%%%%%%lower bound and upper bound
lb = [-66*ones(nc,1); 15*ones(nc,1)];
ub = [66.001*ones(nc,1); 124*ones(nc,1)];
%%%You need to add not touching the big circle



%%%optimization

options = optimoptions('fmincon','Display','iter', 'MaxFunctionEvaluations',10000,'SpecifyObjectiveGradient',true);


disp('Entering Optimization \n')
x = fmincon(fun, X0, A,b, Aeq, beq, lb, ub, nonlcon, options);

fileID = fopen('solution_fmincon_25_hat_sametao.txt','w');
fprintf(fileID,'%20.8f\n',x);
fclose(fileID);