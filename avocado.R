# Loading necessary libraries
library(ggplot2) 
library(tidyverse)
library(dplyr)
library(caret)
library(stats)
library(randomForest)

# Setting data path, loading the data and changing the format to data frame
data_path <- "C:\\Users\\isupe\\OneDrive\\Pulpit\\"            # type yours
avocado <- read.csv(paste0(data_path, "avocado.csv"))
avocado <-as.data.frame(avocado)
head(avocado)

# Describing the data set
dim(avocado)                          # number of rows and columns
str(avocado)                          # describing the column format
summary(avocado)                      # summary of columns
names(avocado)                        # names of columns
avocado$Date <- as.Date(avocado$Date) # changing to Date format

avocado <- avocado[, -1] # dropping index column

# Renaming columns for better understanding
names(avocado) <- c("Date" ,"AveragePrice","Total.Volume", 'Small HASS sold',
                    'Large HASS sold', 'XLarge HASS sold',   "Total.Bags" ,  
                    "Small.Bags"  , "Large.Bags"  , "XLarge.Bags",  "type_organic",      
                    "year","region" )

# Transforming 3 columns into numeric, to use them in correlation
avocado <- transform(avocado, year_n = as.numeric(as.factor(year))) # 4 unique years
avocado <- transform(avocado, region = as.numeric(as.factor(region))) # 54 unique regions
avocado <- transform(avocado, type_organic_n = as.numeric(as.factor(type_organic))) # 1 - conventional, 2 - organic

# Correlation heat map between variables
correlation <- cor(avocado[,c(2,3,4,5,6,7,8,9,10,13,14,15)], avocado[,c(2,3,4,5,6,7,8,9,10,13,14,15)])
heatmap(correlation, main = "Correlation Heatmap")

# Dropping two created columns, as they will not be needed in plots
avocado <- avocado[, -14:-15]

######################################################
# Data Analysis

# Spread of Average Price in the data
ggplot(avocado, aes(x = AveragePrice)) +
  geom_density(fill = "blue", alpha = 0.5) +
  labs(x = "Distribution of Average price", y="Density")

# Type of avocado vs Average Price
ggplot(avocado, aes(x = type_organic, y = AveragePrice, color=type_organic)) + 
  geom_boxplot()+
  scale_y_continuous(limits = c(0,2.5))+
  labs(title="Average Price of one Avocado by type",
       x="Type",y="Average Price",
       color="Type")+ 
  theme_minimal()

# How Average price varies over years
avocado$year<-as.character(avocado$year)
ggplot(avocado, aes(x = year, y = AveragePrice, color=year)) + 
  geom_boxplot()+
  scale_y_continuous(limits = c(0,2.5))+
  labs(title="Average Price of one Avocado by type",
       x="Type",y="Average Price",
       color="Type")+ 
  theme_minimal()

# How average price varies across regions
region_sum <- aggregate(AveragePrice ~ region, avocado, sum)
region_sum <- region_sum[order(-region_sum$AveragePrice), ]
barplot(region_sum$AveragePrice, names.arg = region_sum$region, space=0,
        xlab = "Region", ylab = "Total Average Price",
        main = "Total Average Price by Region", col = "green",
        xlim = c(0, 54),
        ylim = c(0, max(region_sum$AveragePrice) + 100))

########################################################
# Predicting Average Price of Avocado

# Transforming columns to use them in calculations
avocado <- transform(avocado, year = as.numeric(as.factor(year))) # 4 unique years
avocado <- transform(avocado, type_organic = as.numeric(as.factor(type_organic))) # 1 - conventional, 2 - organic

## Using Linear Regression - Model 1

# Creating new data set of 6 columns
avocado_new <- avocado[, c('AveragePrice', 'Total.Volume', 'region', 'type_organic', 
                           'Total.Bags', 'year')]

# Dividing new data set to features and predictor
x <- avocado_new[, c('AveragePrice', 'Total.Volume', 'region', 'Total.Bags', 'year')] # feature columns
y <- avocado_new$type_organic       # predictor variable

# Generating random sample consisting of 80% of rows,
# training and test set
set.seed(1)
train_indices <- sample(nrow(x), nrow(x) * 0.8)
x_train <- x[train_indices, ]
x_test <- x[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

print(paste("X Train Shape", dim(x_train)[1], dim(x_train)[2], sep = ' '))
print(paste("Y Train Shape", length(y_train),  sep = ' '))
print(paste("X Test Shape", dim(x_test)[1],dim(x_test)[2], sep = ' '))
print(paste("Y Test Shape", length(y_test), sep = ' '))

# Standardizing the avocado_new data set
# Creating a new data set called avocado_new_std 
scaler <- scale(avocado_new)
avocado_new_std <- data.frame(scaler)

# Creating vector with features included in regression analysis
feature_cols <- c('Total.Volume', 'region', 'type_organic', 'Total.Bags', 'year')
x <- avocado_new[ , feature_cols]
y <- avocado_new$AveragePrice

# Generating random sample consisting of 80% of rows,
# training and test set
set.seed(1)
split <- sample(nrow(x), nrow(x) * 0.8)
x_train <- x[split, ]
x_test <- x[-split, ]
y_train <- y[split]
y_test <- y[-split]

print(paste('X train shape:', dim(x_train)[1], dim(x_train)[2], sep = ' '))
print(paste('Y train shape:', length(y_train), sep = ' '))
print(paste('X test shape:', dim(x_test)[1], dim(x_test)[2], sep = ' '))
print(paste('Y test shape:', length(y_test), sep = ' '))

# Linear regression, dependent variable is Avarage Price 
linreg1 <- lm(y_train ~ ., data = cbind(y_train, x_train))

# Creating data set eq1 with coefficients
feature_cols <- c('Intercept', 'Total.Volume', 'region', 'type_organic', 'Total.Bags', 'year')
coef <- coef(linreg1)
eq1 <- data.frame(feature_cols, coef)

# Generating predictions based on linear regression 
# for both training and test sets
y_pred_train <- predict(linreg1, newdata = x_train)
y_pred_test <- predict(linreg1, newdata = x_test)


# Model evaluation for Linear Regression Model 1 

# Calculating Root Mean Squared Error between actual values
# and predicted values
RMSE_train <- sqrt(mean((y_train - y_pred_train)^2))
RMSE_test <- sqrt(mean((y_test - y_pred_test)^2))
cat("RMSE for training set is", RMSE_train, "and RMSE for test set is", RMSE_test, "\n")

# Calculating R_squared and Adjusted R_squared
# for training data set...
SS_Residual <- sum((y_train - y_pred_train)^2)
SS_Total <- sum((y_train - mean(y_train))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_train) - 1) / (length(y_train) - ncol(x_train) - 1)
cat("R-squared for train data", r_squared, "and adjusted R-squared for train data", adjusted_r_squared, "\n")
# and for test data set
SS_Residual <- sum((y_test - y_pred_test)^2)
SS_Total <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_test) - 1) / (length(y_test) - ncol(x_test) - 1)
cat("R-squared for test data", r_squared, "and Adjusted R-squared for test data", adjusted_r_squared, "\n")

avocado_full <- avocado[, c('AveragePrice', 'Total.Volume', 'Small.HASS.sold', 'Large.HASS.sold', 'XLarge.HASS.sold',
                            'Total.Bags', 'Small.Bags', 'Large.Bags', 'XLarge.Bags', 'type_organic', 'year', 'region')]


## Using Linear Regression - Model 2

# Standardizing the avocado_new data set
# Creating a new data set called avocado_new_std 
avocado_full_std <- as.data.frame(scale(avocado_full)) #Standardize the features

# Defining feature names
feature_cols <- c('Total.Volume', 'Small.HASS.sold', 'Large.HASS.sold', 'XLarge.HASS.sold', 'Total.Bags', 'Small.Bags',
                  'Large.Bags', 'XLarge.Bags', 'type_organic', 'year', 'region')

# Create the feature matrix x and the target variable y
x <- avocado_full_std[, feature_cols]
y <- avocado_full_std$AveragePrice

# Splitting the data into training and test sets
set.seed(1)  
split <- sample(nrow(x), nrow(x) * 0.8)
x_train <- x[split, ]
x_test <- x[-split, ]
y_train <- y[split]
y_test <- y[-split]

# Print the shapes of the training and test sets
print(paste('X train shape:', dim(x_train)[1], dim(x_train)[2], sep = ' '))
print(paste('Y train shape:', length(y_train), sep = ' '))
print(paste('X test shape:', dim(x_test)[1], dim(x_test)[2], sep = ' '))
print(paste('Y test shape:', length(y_test), sep = ' '))

# Linear regression, dependent variable is Avarage Price 
linreg2 <- lm(y_train ~ ., data = cbind(y_train, x_train))

# Creating data set eq1 with coefficients
feature_cols <- c('Intercept', 'Total.Volume', 'Small.HASS.sold', 'Large.HASS.sold', 'XLarge.HASS sold', 'Total.Bags',
                  'Small.Bags', 'Large.Bags', 'XLarge.Bags', 'type_organic', 'year', 'region')
coef <- coef(linreg2)
eq1 <- data.frame(feature_cols, coef)

# Generating predictions based on linear regression 
# for both training and test sets
y_pred_train <- predict(linreg2, newdata = x_train)
y_pred_test <- predict(linreg2, newdata = x_test)


# Model Evaluation for Linear Regression Model 2

# Calculating Root Mean Squared Error between actual values
# and predicted values
RMSE_train <- sqrt(mean((y_train - y_pred_train)^2))
RMSE_test <- sqrt(mean((y_test - y_pred_test)^2))
cat("RMSE for training set is", RMSE_train, "and RMSE for test set is", RMSE_test, "\n")

# Calculating R_squared and Adjusted R_squared
# for training dataset...
yhat <- predict(linreg2, newdata = x_train)
SS_Residual <- sum((y_train - yhat)^2)
SS_Total <- sum((y_train - mean(y_train))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_train) - 1) / (length(y_train) - dim(x_train)[2] - 1)
cat("R-squared for train data", r_squared, "and adjusted R-squared for train data", adjusted_r_squared, "\n")
# Calculate r-squared and adjusted r-squared for test set
yhat <- predict(linreg2, newdata = x_test)
SS_Residual <- sum((y_test - yhat)^2)
SS_Total <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_test) - 1) / (length(y_test) - dim(x_test)[2] - 1)
cat("R-squared for test data", r_squared, "and Adjusted R-squared for test data", adjusted_r_squared, "\n")


## Predict using Random Forest Regressor

# Create and fit the random forest regression model
model_rf <- randomForest(x_train, y_train, ntree = 100, random_state = 0)

# Make predictions on the training and test sets
y_pred_train <- predict(model_rf, newdata = x_train)
y_pred_test <- predict(model_rf, newdata = x_test)


# Model Evaluation for Random Forest Regressor

# Calculating Root Mean Squared Error between actual values
# and predicted values
RMSE_train <- sqrt(mean((y_train - y_pred_train)^2))
RMSE_test <- sqrt(mean((y_test - y_pred_test)^2))
cat("RMSE for training set is", RMSE_train, "and RMSE for test set is", RMSE_test, "\n")

# Calculating R_squared and Adjusted R_squared
# for training data set...
yhat <- predict(model_rf, newdata = x_train)
SS_Residual <- sum((y_train - yhat)^2)
SS_Total <- sum((y_train - mean(y_train))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_train) - 1) / (length(y_train) - dim(x_train)[2] - 1)
cat("R-squared for train data", r_squared, "and adjusted R-squared for train data", adjusted_r_squared, "\n")
# and for test data set
yhat <- predict(model_rf, newdata = x_test)
SS_Residual <- sum((y_test - yhat)^2)
SS_Total <- sum((y_test - mean(y_test))^2)
r_squared <- 1 - (SS_Residual / SS_Total)
adjusted_r_squared <- 1 - (1 - r_squared) * (length(y_test) - 1) / (length(y_test) - dim(x_test)[2] - 1)
cat("R-squared for test data", r_squared, "and Adjusted R-squared for test data", adjusted_r_squared, "\n")
