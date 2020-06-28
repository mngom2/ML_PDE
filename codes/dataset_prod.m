%%N chebychev points and M time points

%%%change back to -1 1, 1 output and cos pi x


a = -1.; %pi;
b = 1; %pi;
P = 2.0; %*pi;
N_points = 100;
r = (b-a).*rand(N_points,1) + a;
x = r;
%x = a:(b-a)/(N_points-1):b;
%x = 2.0*pi*x/P;
x_plusper = x + P; %b:(2*pi)/N_points:b+ 2*pi;
x_minusper = x - P;
x2 = x+ 2*P;
x3 = x - 2*P;
A = [a,b];
x0 = randi(length(A), size(x));

for i = 1:length(x0)
    if (x0(i)==1)
        x0(i) =a;
    end
    if (x0(i)==2)
        x0(i) = b;
    end
end
% %x =r;
% %x = (sin(r)).^3;
% f = 0;
% ff = [];
% ff1 = ff;
% ff2 = [] ;
% ff3 =ff;
% for i = 1:20
% ff1 = [ff1, -2*(-1)^i/i];
% ff2 = [ff2, 4 * (-1)^i/i^2];
% ff3 = [ff3; (-1)^i*(12/i^3 - 2 *pi^2/i)];
% end
% ff2 = ff2(1:10);
% f = -2*f;
% %x = ff;
% save('data/filtre_train','ff1', 'ff2', 'ff3', 'x')
save('data/input_data','x', 'x_plusper', 'x0','x_minusper', 'x2','x3')
