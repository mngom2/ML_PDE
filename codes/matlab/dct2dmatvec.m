clc; clear all
%pi=3.14;
N=6;

u=inline('(sin(4*x)).*cos(2*y)','x','y');
%chebyshev grid
for i=0:N-1
    xc_cheb(i+1)=-cos((2*i+1)*pi/2/N);
end
[x_cheb2d,y_cheb2d]=ndgrid(xc_cheb,xc_cheb);
%for double precision grid
ux_cheb=u(x_cheb2d,y_cheb2d);
dct_cheb=mirt_idctn(ux_cheb);

const=ones(1,N)*sqrt(2/N);
const(1)=1/sqrt(N);
for i=1:N
    for  j=1:N
        Tk(j,i)=cos((2*i-1)*(j-1)*pi/2/N)*const(j);
    end
end

test=Tk'*ux_cheb*Tk;
max(max(abs(test-dct_cheb)))


% 
% 
% %inefficient
% T2=kron(Tk,Tk);
% 
% ux_cheb2=squeeze(ux_cheb(:,:,1));
% tt=T2*ux_cheb2(:);
% 
% test2=reshape(tt,N,N);
% dct_cheb2=mirt_idctn(ux_cheb2);
% 
% test2=Tk'*ux_cheb2*Tk;
% max(max(abs(test2-dct_cheb2)))


