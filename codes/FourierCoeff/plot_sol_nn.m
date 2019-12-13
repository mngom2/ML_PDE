close all;
clear all;
fileID = fopen('nn.txt','r');
formatSpec = '%f';
u_nn = fscanf(fileID,formatSpec);
fclose(fileID);
x = [-pi:2*pi/100:pi];
plot(x,u_nn, 'bo','MarkerSize',2, 'LineWidth', 3)
 hold on
  u_exact = pi^2/3 * ones(size(x)); % pi/2. * ones(size(x));  %pi^2/3 * ones(size(x)); %zeros(size(x)); % 
%  %u_exact =zeros(size(x)); % pi^2/3 * ones(size(x)); 
  for i = 1:3
   % u_exact = u_exact - 4/(pi*(2*i-1)^2) * cos((2*i-1)*x);
      u_exact = u_exact + 4 * (-1)^i/i^2 * cos(i*x);
%      %u_exact = u_exact + 2 /i * sin(i*x)
  end
  %u_exact = cos(3*x) + sin(7*x); % + x.^2; %sin (30 * x);
  plot(x,u_exact, 'r+','MarkerSize',2, 'LineWidth', 3)
%   hold on
   x = [-pi:2*pi/40:pi];
   plot(x,x.^2, 'g+','MarkerSize',2, 'LineWidth', 3)
% %  hold on
% 
% % hold on
% % u_exact = exp(-abs(x)); %cos(x); %exp(-abs(x)); %x.^4 %cos(x); %(sin(x)).^3;
% % plot(x,u_exact, 'r-','MarkerSize',4)
% % hold on
% u=(1 - exp(-pi)) /pi;
% for i =1:10
%      u = u + 2.0/(pi *(1 + i^2)) *(1 - (-1)^i * exp(-pi))*cos(i*x);
% end
% plot(x, u, 'mo','MarkerSize',4)
%     
