function [u, J] = FNN(WPhi, x)

N = length(WPhi);

w01 = WPhi(1:(N-1)/4);
w12 = WPhi((N-1)/4 + 1 : (N-1)/2);
phi1 = WPhi((N-1)/2 + 1 : (N-1));
phi2 = WPhi(N);
for j = 1:length(w01)
    out(j) = w12(j) * cos(w01(j) * x + phi1(j)) + phi2;
    Jw12_out(j) = cos(w01(j) * x + phi1(j));
    Jw01_out(j) = -w01(j) * sin( w01(j) * x + phi1(j) );
    Jw01_out(j) = -w01(j) * sin( w01(j) * x + phi1(j) );
end
u = sum(out);



end

