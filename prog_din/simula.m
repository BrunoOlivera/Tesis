load ../data/ap_gen_2000_MS_FIXED.csv;

costSamp=[]; volFinal=[];
niveles=0:Ds:VM;
res = 0;
subAgua = 0;
sobreAgua = 0;
for sample=1:T:size(ap_gen_2000_MS_FIXED,2),
  apsmpl=ap_gen_2000_MS_FIXED(:,sample:sample+(T-1));
  aportes=[];
  for anho=1:T, aportes=[aportes; apsmpl(:,anho)]; end;
  volemb=VM/2; costo=0;
  test = [];
  for t=1:52*T,
##    test = [test volemb];
##    volemb
##    aportes(t)
    [val,est]=min(abs(niveles-volemb));
##    est = find(volemb >= niveles)(end);
    if (niveles(est)-volemb > 0)
      sobreAgua += (niveles(est)-volemb)/1000000;
    else
      subAgua += abs(niveles(est)-volemb)/1000000;
    endif
##    (aportes(t)-Dx*(C(est,t)-1))*SegXsem
    costo=costo+cgen(est,C(est,t),0);
    volemb=volemb+(aportes(t)-Dx*(C(est,t)-1))*SegXsem;
##    volemb
    if (volemb > VM)
      res += 1;
    endif
  end
##  res = [res; size(test > VM)(2)];
  costSamp=[costSamp; costo];
  volFinal=[volFinal; volemb];
end;