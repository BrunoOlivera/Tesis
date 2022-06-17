function [costo]=cgen(s,x,esp);

global Dx;

% Turbinado [m3/s]
X=Dx*(x-1);

% Hidráulica
CE=0.19;        % Coeficiente Energético [MW/m3s]
ph=CE*X;        % Potencia hidráulica [MW]
D=350;          % Potenica demandada [MW]
dr=D-ph;        % Demanda residual

% Térmica cara
MT1=250;        % Máximo Técnico [MW]
CT1=4000;       % Rendimiento [USD/MWh]

% Térmica barata
MT2=250;        % Máximo Técnico [MW]
CT2=100;        % Rendimiento [USD/MWh]

if (dr<=MT2),
  costo=dr*24*7*CT2;
else
  costo=MT2*24*7*CT2 + (dr-MT2)*24*7*CT1;
end;