#Imported all the necessary libraries
library(tidyverse)
library(e1071)
library(caret)
library(ggplot2)
library(dplyr)

# I've set the working directory and loaded 'nhis_2022.csv' into data
data <- read.csv("C:/Users/alekh/Downloads/nhis_2022.csv")

# Exploring the data
head(data)
summary(data)
str(data)

# Subsetting the data, taking only adults, i.e., between the ages 18 and 70.
# Converting numeric variables to categoric factors.
data <- data %>%
  filter(AGE >= 18, AGE <= 70, STROKEV %in% c(1,2)) %>%
  mutate(
    Sex = factor(SEX, levels = c(1,2), labels = c("Male","Female")),
    Stroke = factor(STROKEV, levels = c(1,2), labels = c("No","Yes"))
  )


# Renaming variables to clear column names.
names(data)[names(data)=="AGE"] <- "Age"
names(data)[names(data)=="HRSLEEP"] <- "Hours Of Sleep"
names(data)[names(data)=="HOURSWRK"] <- "Hours Worked"
names(data)[names(data)=="ALCDAYSYR"] <- "Alcohol Consumption Days Per Year"
names(data)[names(data)=="CIGDAYMO"] <- "Cigarettes Consumed Per Month"
names(data)[names(data)=="MOD10DMIN"] <- "Duration Of Moderate Activity(in mins)"
names(data)[names(data)=="VIG10DMIN"] <- "Duration Of Vigorous Activity(in mins)"

# Predicting the stroke status (Yes/No) in adults (18–70) 
# using SVMs on predictors:
# Age, Sex, Hours Of Sleep, Hours Worked,Alcohol Consumption Days
# Cigarettes/Month, Moderate & Vigorous Activity (mins).


# Cleaning the invalid codes and then replace those with NA, 
#lastly, drop those null values.
codes <- c(996, 997, 998, 999)
variables <- c("Age", "Hours Of Sleep", "Hours Worked",
               "Alcohol Consumption Days Per Year", 
               "Cigarettes Consumed Per Month",
               "Duration Of Moderate Activity(in mins)", 
               "Duration Of Vigorous Activity(in mins)")
# Keeping both Moderate and Vigorous activity 
#since they are not highly correlated.
data <- data %>%
  mutate(across(all_of(variables), ~ ifelse(. %in% codes, NA, .))) %>%
  na.omit()

#Exploratory Data Analysis
# Histogram of Age distribution by Age
ggplot(data, aes(x = Age, fill = Stroke)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  labs(
    title = "Age Distribution by Stroke Status",
    fill  = "Stroke"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position   = "top"
  )

# A density plot of sleeping hours by Stroke
ggplot(data, aes(x = `Hours Of Sleep`, fill = Stroke)) +
  geom_density(alpha = 0.6, color = NA) +
  coord_cartesian(xlim = c(0, 13)) +
  labs(
    title = "Distribution of the Sleep Hours by Stroke",
    fill  = "Stroke"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "top"
  )

# Computing correlation between moderate & vigorous activity
correlation <- cor(
  data$`Duration Of Moderate Activity(in mins)`,
  data$`Duration Of Vigorous Activity(in mins)`
)
print(paste("Correlation (r) =", round(correlation, 5)))

# Plotting the above
ggplot(data, aes(
    x = `Duration Of Moderate Activity(in mins)`,
    y = `Duration Of Vigorous Activity(in mins)`
  )) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "lm", se = FALSE,
              color = "green") +
  labs(
    title = "Moderate Activity vs Vigorous Activity (mins)",
  ) +
  theme_minimal()

# Scaling the variables variables
data[variables] <- scale(data[variables])

# Splitting the data into train and test sets.
set.seed(42)
train_data <- createDataPartition(data$Stroke, p = 0.7, list = FALSE)
train_set <- data[train_data, ]    
test_set <- data[-train_data, ] 

# Increased 'Yes' class weight to 50 
#to force the model to better recognize 
#minority class during training.
# Chosen this setting of balance, to address the imbalance
# while avoiding extremes. This ratio improved 
# stroke detection without excessive false alarms.
weights <- c("No" = 1, "Yes" = 50)

#PART 1: Linear SVM 
set.seed(42)
tune_one <- tune(svm,
                 Stroke ~ `Age` + Sex + `Hours Of Sleep` + `Hours Worked` 
                 + `Alcohol Consumption Days Per Year` 
                 + `Cigarettes Consumed Per Month`
                 + `Duration Of Moderate Activity(in mins)` 
                 + `Duration Of Vigorous Activity(in mins)`,
                 data          = train_set, kernel        = "linear",
                 ranges        = list(cost = c(0.01, 0.1, 1)), 
                 class.weights = weights)

# Choosing the best linear svm from tuning
svm_one <- tune_one$best.model

# Prediction and evaluation metrics
prediction_one <- predict(svm_one, test_set)
confusion_mat_one <- confusionMatrix(prediction_one, test_set$Stroke)
print(confusion_mat_one)
precision_one <- posPredValue(prediction_one, test_set$Stroke, positive = "Yes")
recall_one <- sensitivity(prediction_one,  test_set$Stroke, positive = "Yes")
f1_one <- 2 * (precision_one * recall_one) / (precision_one + recall_one)
accuracy_one <- confusion_mat_one$overall["Accuracy"]
precision_one
recall_one
f1_one 
accuracy_one

# Extracting variable importance from the fitted linear SVM model
weight_vector <- as.numeric(t(svm_one$coefs) %*% svm_one$SV)
names(weight_vector) <- colnames(svm_one$SV)

#'importance_df' consists ofabsolute weights and sort them in decreasing order
importance_df <- data.frame(
  Variable = names(weight_vector),
  Importance = abs(weight_vector)
)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

#Plotting the variable importance
ggplot(importance_df, aes(
  x = reorder(Variable, Importance),
  y = Importance
)) +
  geom_col(fill = "#619CFF") +
  coord_flip() +
  labs(
    title = "Variable Importance (Linear SVM)",
    x     = NULL,
    y     = "Importance"
  ) +
  theme_minimal(base_size = 13)

ggsave("variable_imp.png", width = 5, height = 5, dpi = 350)

#PART 2: Radial SVM  
tune_two <- tune(svm,
                 Stroke ~ `Age` + Sex + `Hours Of Sleep` + `Hours Worked` 
                 + `Alcohol Consumption Days Per Year` 
                 + `Cigarettes Consumed Per Month`
                 + `Duration Of Moderate Activity(in mins)` 
                 + `Duration Of Vigorous Activity(in mins)`,
                 data          = train_set,
                 kernel        = "radial",
                 ranges        = list(cost = c(0.1, 1), gamma = c(0.01, 0.1)),
                 class.weights = weights)

# Choosing the best radial svm from tuning
svm_two <- tune_two$best.model
svm_two

# Prediction and evaluation metrics
prediction_two <- predict(svm_two,    test_set)
confusion_mat_two <- confusionMatrix(prediction_two, test_set$Stroke)
print(confusion_mat_two)
precision_two <- posPredValue(prediction_two, test_set$Stroke, positive = "Yes")
recall_two<- sensitivity(prediction_two,  test_set$Stroke, positive = "Yes")
f1_two<- 2 * (precision_two * recall_two) / (precision_two + recall_two)
accuracy_two<- confusion_mat_two$overall["Accuracy"]
precision_two
recall_two
f1_two 
accuracy_two

#PART 3: Polynomial SVM 
tune_three <- tune(svm,
                   Stroke ~ `Age` + Sex + `Hours Of Sleep` + `Hours Worked` 
                   + `Alcohol Consumption Days Per Year` 
                   + `Cigarettes Consumed Per Month`
                   + `Duration Of Moderate Activity(in mins)` 
                   + `Duration Of Vigorous Activity(in mins)`,
                   data          = train_set, kernel        = "polynomial",
                   ranges        = list(cost   = c(0.1,1), degree = c(3,4),
                                        coef0  = c(0.5,1)),
                   class.weights = weights)

# Choosing the best polynomial svm from tuning
svm_three <- tune_three$best.model
svm_three

# Prediction and evaluation metrics
prediction_three<- predict(svm_three,    test_set)
confusion_mat_three  <- confusionMatrix(prediction_three, test_set$Stroke)
print(confusion_mat_three)
precision_three<- posPredValue(prediction_three, test_set$Stroke, positive = "Yes")
recall_three<- sensitivity(prediction_three,  test_set$Stroke, positive = "Yes")
f1_three <- 2 * (precision_three * recall_three) / (precision_three + recall_three)
accuracy_three  <- confusion_mat_three$overall["Accuracy"]
precision_three
recall_three 
f1_three 
accuracy_three 

# Comparing the accuarcy results by plotting
model_results <- data.frame(
  Model = c("Linear SVM", "Radial SVM", "Polynomial SVM"),
  Accuracy = c(accuracy_one, accuracy_two, accuracy_three)
)

ggplot(model_results, aes(x = Model, y = Accuracy)) +
  geom_col() +
  ylim(0, 1) +
  labs(title = "SVM Model Accuracies", x = "Model", y = "Accuracy") +
  theme_minimal()

#From the above plot, polynomial model has performed the highest, 
#about 73 % , radial performs next best , 
#and then linear svm performs good, does purely linear separation. 


#RESULTS: The linear SVM, with best cost 1 , has 8739 support vectors, 
#suggesting high complexity, and it has an accuracy of 70%. 
#It is poor at detecting strokes, precision is 5.1%; 

#The radial SVM, with best cost 1 and gamma 0.1 , 
#has 7352 support vectors,so its efiicient, 
#and it has an accuracy of 70.89%. 
#It is slightly better than linear svm, 
#but still poor at detecting strokes, precision is 5.21%;

#The polynomial SVM is the best among all, 
#with best cost 1, degree 4, and coef0 0.5 , has 6850 support vectors, 
#and it has an accuracy of 72.66%. 
#It is still suffers from low precision like the rest of the two models,
#for "Yes" , but has better F1-score.

#The minority class is "yes" stroke and 
#all the three models perform poorly in this one.
