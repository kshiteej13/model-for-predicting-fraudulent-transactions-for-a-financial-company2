library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(caret)
library(matrixStats)

data <- read_csv("Fraud.csv")

set.seed(0)

size <- dim(data[data$isFraud == 1,])[1]
temp_df_fraud <- data[data$isFraud == 1,]
temp_df_not_fraud <- data[data$isFraud == 0,][sample(seq(1, size), size),]

df <- full_join(temp_df_not_fraud, temp_df_fraud)

rm(temp_df_fraud, temp_df_not_fraud)

df_num <- df %>%
  select(-c(nameOrig, nameDest, type))
head(df_num)

set.seed(0)

test_index <- createDataPartition(df_num$isFraud, times = 1, p = 0.2, list = FALSE) # first create the indexes for the test set

test_x <- select(df_num, -isFraud)[test_index,]
test_y <- df_num$isFraud[test_index]

train_x <- select(df_num, -isFraud)[-test_index,]
train_y <- df_num$isFraud[-test_index]

# change de training data as factor because we will only have to values

train_y <- as.factor(train_y)

# Model K Means

predict_kmeans <- function(x, k) {
    centers <- k$centers    # extract cluster centers
    # calculate distance to cluster centers
    distances <- sapply(1:nrow(x), function(i){
                        apply(centers, 1, function(y) dist(rbind(x[i,], y)))
                 })
  max.col(-t(distances))
        }

set.seed(0)

k <- kmeans(train_x, centers = 2)
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, 0, 1)

results <- data_frame(method = "K means", accuracy = mean(kmeans_preds == test_y)) # save the results in the data frame

set.seed(0)

train_glm <- train(train_x, train_y,
                     method = "glm")

glm_preds <- predict(train_glm, test_x)

results <- bind_rows(results, # add accuracy to the df
                          data_frame(method="Logistic regression",
                                     accuracy = mean(glm_preds == test_y)))

set.seed(0)

train_lda <- train(train_x, train_y,
                     method = "lda")

lda_preds <- predict(train_lda, test_x)

results <- bind_rows(results, # add accuracy to the df
                          data_frame(method="LDA",
                                     accuracy = mean(lda_preds == test_y)))

set.seed(0)

train_knn <- train(train_x, train_y,
                     method = "knn",
                     tuneGrid = data.frame(k = seq(1.95, 2, 0.01)))

knn_preds <- predict(train_knn, test_x)

results <- bind_rows(results, # add accuracy to the df
                          data_frame(method="KNN",
                                     accuracy = mean(knn_preds == test_y),
                                     tune = train_knn$bestTune %>% pull()))

set.seed(0)

train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = data.frame(mtry = seq(5.4,5.6,0.1)),
                  importance = TRUE)

rf_preds <- predict(train_rf, test_x)

results <- bind_rows(results, # add accuracy to the df
                          data_frame(method="RF",
                                     accuracy = mean(rf_preds == test_y),
                                     tune = train_rf$bestTune %>% pull()))

models <- matrix(c(glm_preds, knn_preds, rf_preds), ncol = 3)

ensemble_preds <- ifelse(rowMedians(models) == 1, 0, 1)

results <- bind_rows(results,
                          data_frame(method="Ensemble",
                                     accuracy = mean(ensemble_preds == test_y)))


results %>% knitr::kable()


