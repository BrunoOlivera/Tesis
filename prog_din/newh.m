Q=[0.2 0.8 0 0;
  0.2 0.3 0.5 0;
  0 0.5 0.3 0.2;
  0 0 0.8 0.2];

M=zeros(size(Q));

for i=1:4,
  M(i,1)=Q(i,1);
  for j=2:4, M(i,j)=M(i,j-1)+Q(i,j); end
end

function [hp]=newh(M,h);
r=rand();
idx=find ((r<=M(h,:)) & (r<M(h,:)));
hp=idx(1);
