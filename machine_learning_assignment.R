library(ggplot2)
library(MASS)
library(mda)
library(mvtnorm)
library(e1071)
library(caTools)
library(ROCR) 
#SVM MODEL
colnames(processed_email) <- c("text", "y", "x1")
model = svm(y ~ x1, data = processed_email)
y_pred_svm <- predict(model, newdata = processed_email)
processed_email$count = c(1:5787)
processed_email$y_pred_svm = as.character(y_pred_svm)
num_data = as.data.frame(cbind(processed_email,c(1:5787)))
ggplot(data = num_data)+geom_point(aes(count, x1, color = y_pred_svm))

#LDA model
lda.model = lda(y ~ x1,data = processed_email)
y_pred <- predict(lda.model, newdata = processed_email)$class
processed_email$Y_lda_prediction <- as.character(y_pred)
processed_email$Y_actual_pred <- paste(processed_email$X2, processed_email$Y_lda_prediction, sep=",")
ggplot(data = processed_email)+geom_point(aes(count,x1, color = Y_actual_pred))

#LOGISTIC REGRESSION
logistic_model_quadgram <- glm(y~x1,data = processed_email_quadgram, family = "binomial")
predict_reg_quadgram <- predict(logistic_model_quadgram, newdata = processed_email_quadgram, type = "response")
predict_reg_quadgram <- ifelse(predict_reg_quadgram >0.5, 1, 0)
processed_email_quadgram$count = c(1:4999)
processed_email_quadgram$y_pred_logistic_quadgram = as.character(predict_reg_quadgram[1:4999])
processed_email_quadgram$y_pair_quadgram = paste(processed_email_quadgram$y, processed_email_quadgram$y_pred_logistic_quadgram, sep=",")
ggplot(data = processed_email_quadgram)+geom_point(aes(count,x1, color = y_pair_quadgram))