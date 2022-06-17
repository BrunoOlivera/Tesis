function [ap]=aprt(esp);
r=rand();
g=@(y) y + ((1+y).*exp(-y)+r-1)./(y.*exp(-y));
y=1; for k=1:10, y=g(y); end;
ap=esp*y/2;
