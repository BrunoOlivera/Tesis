global N M Dx Ds;
T=2;    % Horizonte de planificación en años (52sem/año).
N=101;  % Número de estados para el nivel del embalse
        % s_1=0,...,s_101=VM.
M=11;   % Discretización del control a aplicar en cada paso
        % c_0=0,...,c_11=TM.

% DATOS UNIDAD HIDRÁULICA
VM=8200*100^3;  % Volúmen máximo del embalse en m3.
TM=680;         % Turbinado máximo en m3/s.
SegXsem=3600*24*7;

Ds=VM/(N-1);    % Diferencia de volumen entre estados del lago.
Dx=TM/(M-1);    % Diferencia entre nivel de turbinado (control).

% VECTOR DE APORTES ESPERADOS SEMANALES
ANIOS_SAMPLE = 20;
sample_aps = gen_aportes(ANIOS_SAMPLE);
EST_1_MEAN = mean(sample_aps(1:13,:)(:)');
EST_2_MEAN = mean(sample_aps(14:26,:)(:)');
EST_3_MEAN = mean(sample_aps(27:39,:)(:)');
EST_4_MEAN = mean(sample_aps(40:52,:)(:)');

##EST_1_APS = sample_aps(1:13,:)(:)';
##EST_2_APS = sample_aps(14:26,:)(:)';
##EST_3_APS = sample_aps(27:39,:)(:)';
##EST_4_APS = sample_aps(40:52,:)(:)';

##aptsan=[154.3*ones(13,1); 403.2*ones(13,1); 537.4*ones(13,1); 320.2*ones(13,1)];
aptsan=[EST_1_MEAN*ones(13,1); EST_2_MEAN*ones(13,1); EST_3_MEAN*ones(13,1); EST_4_MEAN*ones(13,1)];
aportes=[];
for t=1:T, aportes=[aportes; aptsan]; end;

% MATRIZ DE COSTOS FUTUROS
A=zeros(N,52*T+1);
C=zeros(N,52*T);

for t=52*T:-1:1,
  for s=1:N,
    fBllmn=zeros(M,1);
    for x=1:M,
      if (Ds*(s-1)>=Dx*(x-1)*SegXsem),
##        prbfn=dinm(s,x,aportes(t));
##        if idivide(max(0,t-52)-1,13,"floor") == 0:
##            prbfn=dinm2(s,x,EST_1_APS);
##        elseif idivide(max(0,t-52)-1,13,"floor") == 1:
##            prbfn=dinm2(s,x,EST_2_APS);
##        elseif idivide(max(0,t-52)-1,13,"floor") == 2:
##            prbfn=dinm2(s,x,EST_3_APS);
##        elseif idivide(max(0,t-52)-1,13,"floor") == 3:
##            prbfn=dinm2(s,x,EST_3_APS);
        prbfn=dinm2(s,x,sample_aps(mod(t-1,52)+1,:));
        fBllmn(x)=cgen(s,x,aportes(t))+sum(prbfn.*A(:,t+1));
      else
        fBllmn(x)=inf;
      end
    end;
    [val,idx]=min(fBllmn);
    A(s,t)=val;
    C(s,t)=idx;
  end;
end;