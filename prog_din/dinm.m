function [prbfn]=dinm(s,x,esp);

global N M Dx Ds;

% Turbinado [m3/s]
X=Dx*(x-1);

% Embalse [m3]
S=Ds*(s-1)-X*(3600*24*7);

limsEmb=[0 Ds/2:Ds:(N-1)*Ds];
##limsEmb=[0 Ds:Ds:(N-1)*Ds];
limsEmb=[limsEmb limsEmb(end)*100];
if (S<0)
  idx=1;
else
  idx=find(S>=limsEmb);
  idx=idx(end);
end;

a=2/esp;
prbfn=zeros(N,1);
for i=idx:(length(limsEmb)-2);
  p=(limsEmb(i+1)-S)/(3600*24*7);
  q=(limsEmb(i+2)-S)/(3600*24*7);
  prbfn(i+1)=(1+a*p)*exp(-a*p)-(1+a*q)*exp(-a*q);
end
prbfn(idx)=1-sum(prbfn);
