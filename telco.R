data <- read.csv("Tlc.csv")[2:42]
head(data)

lr <- glm(Churn~tenure+PaperlessBilling+TotalCharges+MultipleLines_No+Contract_Month.to.month+Contract_OneYear+PaymentMethod_ElectronicCheck, data = data, family = "binomial")
summary(lr)

train <- read.csv('train.csv')
test <- read.csv('test.csv')

lr <- glm(Churn~tenure+PaperlessBilling+TotalCharges+MultipleLines_No+Contract_Month.to.month+Contract_OneYear+PaymentMethod_ElectronicCheck, data = train, family = "binomial")
summary(lr)

pred <- predict(lr, newdata = test, type="response")

final <- rep(0, times = length(pred))
final[pred > 0.2] <- 1

addmargins(table(test$Churn, final))
rsum(final[test$Churn == 0])

