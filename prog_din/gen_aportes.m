function [aps]=gen_aportes(ANIOS)
 
  Ehe=[23.1 60.4 80.5 47.9;
      154.3 403.2 537.4 320.2;
      308.6 806.3 1074.9 640.4;
      780.5 2039.7 2719.0 1620.0];
  aps = [];
##  ANIOS = 20;
  H = 2;  # MS
  for anio=1:ANIOS
    ap = [];
    for semana=1:52
      est = idivide(semana,13,"ceil");
      ap = [ap; aprt(Ehe(H,est))];
    end;
    aps = [aps ap];
  end;
