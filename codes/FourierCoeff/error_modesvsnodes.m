close all;
clear all;
n_nodes = 1:15;
modes = 1:5;
base_name='errmode_';
err = zeros(n_nodes(end), modes(end));
for j = modes
    file_name=[base_name,num2str(j),'.txt'];
    fileID = fopen(file_name,'r');
    formatSpec = '%f';
    err(:,j) = fscanf(fileID,formatSpec);
    fclose(fileID);

    loglog(n_nodes, err(:,j),'+', 'MarkerSize',7, 'LineWidth',3)
    hold on
 
end
%pcolor(modes, 1:5, err(1:5,:))
hold off