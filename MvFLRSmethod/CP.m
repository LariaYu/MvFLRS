function Ve=CP(CoL)
[Ws]=W_CP_tensor(CoL);
Ve=exp(Ws.V.^2);
Vl=log(Ws.V.^2);
end
