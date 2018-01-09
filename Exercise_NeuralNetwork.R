#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/
set.seed(500)
library(MASS)
data <- Boston

apply(data, 2, function(x) sum(is.na(x)))

# glm
index <- sample(1:nrow(data), round(0.75*nrow(data)))
train <- data[index,]
test <- data[-index,]
lm.fit <- glm(medv ~.,data = train)
summary(lm.fit)
pr.lm <- predict(lm.fit, test)

# Mean Squared Error
# 均方误差是指参数估计值与参数真值之差平方的期望值;
# MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
MSE.lm <- sum((pr.lm - test$medv)^2)/nrow(test)


# scale and split the data
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))

train_ <- scaled[index,]
test_ <- scaled[-index,]
# Note that scale returns a matrix that needs to be coerced into a data.frame.



#install.packages("neuralnet")
library(neuralnet)
n <- names(train_)
                                      # n中除了medv的其他所有
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train_,hidden=c(5,3),linear.output=T)
# Note:
# For some reason the formula y~. is not accepted in the neuralnet() function. 
# You need to first write the formula and then pass it as an argument in the fitting function.

# The hidden argument accepts a vector with the number of neurons for each hidden layer, 
# while the argument linear.output is used to specify 
# whether we want to do regression linear.output=TRUE or classification linear.output=FALSE


plot(nn)



## Predicting medv using the neural network
# compute returns:
# neurons: a list of the neurons' output for each layer of the neural network.
# net.result: a matrix containing the overall result of the neural network.
pr.nn <- compute(nn,test_[,1:13])

pr.nn_ <- pr.nn$net.result * (max(data$medv)-min(data$medv)) + min(data$medv)
test.r <- (test_$medv) * (max(data$medv)-min(data$medv)) + min(data$medv)
MSE.nn <- sum((test.r - pr.nn_)^2) / nrow(test_)


print(paste(MSE.lm,MSE.nn))



## cross validation
#通过设定函数par()的各个参数来调整我们的图形
#mfrow用于设定图像设备的布局（简单的说就是将当前的绘图设备分隔成了nr*nc个子设备）
par(mfrow=c(1,2))

plot(test$medv, pr.nn_, col='red', main='Real vs predicted NN', pch=18, cex=0.7) # pch: 点的样式    cex：指定符号的大小. cex是一个数值，表示pch的倍数，默认是1.5倍
abline(0,1,lwd=2) #属性值：col:线条的颜色 lty:线条的类型 lwd:线条的宽度
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test$medv,pr.lm,col='blue',main='Real vs predicted lm',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='LM',pch=18,col='blue', bty='n', cex=.95)

plot(test$medv,pr.nn_,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test$medv,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))
# we can see that the predictions made by the neural network are (in general) more concetrated around the line 
# (a perfect alignment with the line would indicate a MSE of 0 and thus an ideal perfect prediction)
# than those made by the linear model.




## cross validation
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



