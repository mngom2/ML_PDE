clc; clear all; close all
%pi=3.14;
N=8;

u=inline('(sin(2*x))+2','x');
N=32;
u=inline('(sin(8*x))+2','x');
%chebyshev grid
% for i=0:N-1
%     xc_cheb(i+1)=-cos((2*i+1)*pi/2/N);
% end
xc_cheb=linspace(-1,1,N);

%for double precision grid
ux_cheb=u(xc_cheb);
dct_cheb=mirt_idctn(ux_cheb);

const=ones(1,N)*sqrt(2/N);
const(1)=1/sqrt(N);
for i=1:N
    for  j=1:N
        Tk(j,i)=cos((2*i-1)*(j-1)*pi/2/N)*const(j);
    end
end

modes=Tk*ux_cheb';
max(max(abs(modes-dct_cheb)))
% 
% for i=1:N
%     if modes(i)<1e-5
%         modes(i)=0;
%     end
% end

modes(end-5:end)=0;

testinv=Tk'*modes;
max(max(abs(testinv-ux_cheb')))

subplot(1,2,1)
plot(xc_cheb,ux_cheb,'-*')
hold on
plot(xc_cheb,testinv,'r-o')
subplot(1,2,2)
semilogy(abs(modes),'ko','MarkerSize', 10)
%semilogy(sort(abs(test),'descend'),'ko','MarkerSize', 10)

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


