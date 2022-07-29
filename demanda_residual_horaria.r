#datos <- read.csv("C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_ML_FIXED.csv", header = TRUE, sep = ';')
datos <- read.csv("C:/Users/D255728/Documents/Prototipo/data/demanda_residual_horaria.csv", header = TRUE, , sep = ';')

#fit <- auto.arima(datos$Demanda[0:3287], max.p = 50, max.q = 50, max.d = 5)
fit <- auto.arima(datos$DEMANDA_RESIDUAL[0:35064], max.p = 50, max.q = 50, max.d = 5)


# plot(datos$DEMANDA_RESIDUAL_POSITIVA, type = "l", lwd = 2)
plot(datos$DEMANDA_RESIDUAL, type = "l", lwd = 2)
lines(fit$fitted, type = "l", col = "red", , lwd = 2)

plot(datos$DEMANDA_RESIDUAL[35064:43824], type = "l", lwd = 2)
preds <- forecast(fit,8760)
lines(preds$mean, type = "l", col = "red", , lwd = 2)

plot(preds$mean)


d2 <- datos
d2$arima <- fit$fitted

#write.csv(d2, "C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_con_ARIMA.csv")



plot(datos$DEMANDA_RESIDUAL_POSITIVA, type = "l", lwd = 2)
#lines(fit$fitted, type = "l", col = "red", , lwd = 2)
lines(Mod(fft(datos$DEMANDA_RESIDUAL_POSITIVA)), col = "red", , lwd = 2)

d2 <- datos[0:3287,]
d2$arima <- fit$fitted
d3 <- tail(datos, n=365)
d3$arima <- fit$fitted



#write.csv(d2, "C:/Users/D255728/Documents/ProyectoGAN/datos/datos_diarios_ML_FIXED_TRAIN.csv")


tmp_fit <- fit
tmp_data <- datos$Demanda[0:3287]
pred <- c()
for (i in seq(365)) {
  y <- forecast(tmp_fit,1)$mean[1]
  tmp_data <- append(tmp_data, y)
  pred <- append(pred, y)
  tmp_fit <- arima(tmp_data, c(20,1,0))
  
}
plot(real_2019, type = "l", lwd=2, col = "blue")
lines(pred, type = "l", lwd=2, col = "red")
#lines(seq(365), predicted_2019$mean, type = "l", lwd=2, col = "red")