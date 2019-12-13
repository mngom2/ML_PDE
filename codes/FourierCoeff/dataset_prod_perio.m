%%N chebychev points and M time points




a =-1.0;
b = 1.0;
r = (b-a).*rand(100,1) + a;
P = 2.0;
%x = r;
x = a: (b-a)/99.0:b;
x_per = 2*pi * x / P;
%x_per = 2*pi*(a:(b-a)/20:b)/(pi/15);
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
save('data/funlog_per','x', 'x0', 'x_per')
