%%N chebychev points and M time points




a = -pi;
b = pi;
r = (b-a).*rand(100,1) + a;
%x = r;
%x = a:(b-a)/100:b;
x =r;
%x = (sin(r)).^3;
f = 0;
ff = [];
ff1 = ff;
ff2 = [] ;
ff3 =ff;
for i = 1:20
ff1 = [ff1, -2*(-1)^i/i];
ff2 = [ff2, 4 * (-1)^i/i^2];
ff3 = [ff3; (-1)^i*(12/i^3 - 2 *pi^2/i)];
end
ff2 = ff2(1:10);
f = -2*f;
%x = ff;
x = (1+1).*rand(20,1) -1;
%save('data/filtre_train','ff1', 'ff2', 'ff3', 'x')
save('data/filtre_train_x','x', 'ff1')
