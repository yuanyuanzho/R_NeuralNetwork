#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
set.seed(500)
loan_data <- read.csv(file = "/Users/eavy/Downloads/loan.csv")

apply(loan_data, 2, function(x) sum(is.na(x)))

# glm
index <- sample(1:nrow(loan_data), round(0.75*nrow(loan_data)))
train <- loan_data[index,]
test <- loan_data[-index,]
# recode loan_data$Decision =="2" as TRUE, else as FALSE
loan_data$Decision = loan_data$Decision =="2"
lm.fit <- glm(Decision ~.,data = loan_data)
summary(lm.fit)
pr.lm <- predict(lm.fit, test)
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test) # 20.45375856

# remove non-numeric columns
#mydata <- loan_data[, sapply(loan_data, is.numeric)]
col <- c(2,3,8,9,12,13)
maxs <- apply(loan_data[,col], 2, max) 
mins <- apply(loan_data[,col], 2, min)
scaled <- as.data.frame(scale(loan_data[,col], center = mins, scale = maxs - mins))

train_ <- scaled[index,]
#head(train_)
test_ <- scaled[-index,]


#install.packages("neuralnet")
library(neuralnet)
n <- names(train_)
f <- as.formula(paste("Decision ~", paste(n[!n %in% "Decision"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
plot(nn)

pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test_$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(test_)


print(paste(MSE.lm,MSE.nn))



par(mfrow=c(1,2))

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test$medv,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))

library(boot)
set.seed(200)
lm.fit <- glm(medv~.,data=data)
cv.glm(data,lm.fit,K=10)$delta[1]

set.seed(450)
cv.error <- NULL
k <- 10

library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(data),round(0.9*nrow(data)))
  train.cv <- scaled[index,]
  test.cv <- scaled[-index,]
  
  nn <- neuralnet(f,data=train.cv,hidden=c(5,2),linear.output=T)
  
  pr.nn <- compute(nn,test.cv[,1:13])
  pr.nn <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
  
  test.cv.r <- (test.cv$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
  
  cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)
  
  pbar$step()
}

mean(cv.error)
cv.error
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)



