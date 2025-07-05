# Laden und Überblick
data(iris)
set.seed(42)
# Split in Trainings- und Testdaten (80/20)
idx <- sample(1:nrow(iris), size = 0.8 * nrow(iris))
train <- iris[idx, ]
test  <- iris[-idx, ]

library(randomForest)
model_rf <- randomForest(Species ~ ., data = train)
pred_rf  <- predict(model_rf, test)
table(pred_rf, test$Species)