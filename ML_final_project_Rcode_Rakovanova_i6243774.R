# Machine Learning Final Project
# Author: Barbora Rakovanová (i6243774)
# 6.6.2024

library(tree)
library(caret)
library(gbm)
library(randomForest)
library(dplyr)
library(ggplot2)
library(pROC)
library(kernelshap)
library(shapviz)
library(shapper)
library(xgboost)

######################## Data Loading and Cleaning ###########################
data_matches <- read.csv("atp_matches_till_2022.csv", 
                         header = TRUE, 
                         sep = ",", 
                         dec = ".")

data_matches <- data_matches %>%
  mutate(tourney_date = as.Date(as.character(tourney_date), format = "%Y%m%d"))

#removing walkovers and retirements (removes 4891 entries)
data_matches <- data_matches[!grepl("W/O|RET", data_matches$score), ]


####################### Feature Extraction ##################################
# environmental feature: equal for both players
# players feature: different for each player

data_matches$row_number <- c(1:nrow(data_matches))
data_matches_1991 <- data_matches[data_matches$tourney_date >= "1991-01-01", ]
data_matches <- arrange(data_matches, tourney_date)


# ---------------------------- Home Advantage ---------------------------------
#removing Davis Cup matches
data_matches_no_Davis_1991 <- data_matches_1991[data_matches_1991$tourney_level != "D", ]
tourney_name_unique <- unique(data_matches_no_Davis_1991$tourney_name)

data("world.cities")

#function to associate the tournament city with a country
city_to_country <- function(city_name) {
  city_info <- world.cities[grep(tolower(city_name), tolower(world.cities$name)), ]
  if (nrow(city_info) == 0) {
    return("Unknown")
  } else {
    return(unique(city_info$country.etc))
  }
}

tour_country <- c()

for(c in 1:length(tourney_name_unique)){
  tour_country[c] <- city_to_country(tourney_name_unique[c])
}

tour_place <- data.frame(City = tourney_name_unique, Country = tour_country)

#several tournaments are not called by the city name, resulting in "Unknown"
unknown_indices <- which(tour_country == "Unknown")

#manually assigning the country of "Unknown" entries
unknown_countries <- c("Australia", "Australia", "Germany", "USA", "USA",
                       "Hong Kong", "Japan", "Monaco", "Germany", "Italy", 
                       "France", "Great Britain", "Netherlands", "Great Britain", 
                       "Sweden", "Switzerland", "Germany", "Canada", "Netherlands", 
                       "USA", "USA", "USA", "Australia", "Japan", "Sweden", 
                       "France", NA, NA, "Spain", "Malaysia", "Malaysia", "USA", 
                       "USA", "Austria", "Russia", "Bermuda", "Portugal", 
                       "Germany", "India", "USA", "Germany", "Netherlands",
                       "Spain", NA, "Australia", "Brazil", "Spain", "Greece",
                       "Austria", "China", "China", "Great Britain", "Morocco",
                       "Brazil", NA, NA, NA, "Brazil", "USA", "Germany", "Italy",
                       "Russia", "Germany", "Kazakhstan", "Japan", "Australia",
                       "Australia", "Serbia", "Australia", "Australia", "Serbia")

i = 0
for(u in unknown_indices){
  i <- i+1
  tour_place$Country[u] <- unknown_countries[i]
}

#several places are incorrectly assigned:
# places that were falsely assigned:
# 6 USA, 18 USA, 21 France, 31 Croatia,  38 Italy, 45 USA, 
# 63 Italy, 67 Germany, 81 New Zealand, 92 Qatar, 98 Spain, 106 Chile, 
# 109 Costa Rica, 122 Spain, 134 USA, 137 Great Britain, 144 Great Britain, 
# 147 Poland, 150 USA, 160 USA, 172 Ecuador, 174 Bulgaria, 201 USA, 205 USA
# + change all UK to Great Britain

#finding falsely assigned countries and reassigning them correctly
tour_place$Country[175] <- "Morocco"

false_indices <- c(6, 18, 21, 31, 38, 45, 63, 67, 81, 92, 98, 106, 109, 122, 
                   134, 137, 144, 147, 150, 160, 172, 174, 201, 205)

false_countries <- c("USA", "USA", "France", "Croatia", "Italy", "USA", "Italy",
                     "Germany", "New Zealand", "Qatar", "Spain", "Chile",
                     "Costa Rica", "Spain", "USA", "Great Britain",
                     "Great Britain", "Poland", "USA", "USA", "Ecuador",
                     "Bulgaria", "USA", "USA")

i = 0
for(f in false_indices){
  i <- i+1
  tour_place$Country[f] <- false_countries[i]
}

#replacing UK by Great Britain (to follow convention from data_matches)
for(k in 1:length(tour_place$Country)){
  if(!is.na(tour_place$Country[k]) && tour_place$Country[k] == "UK"){
    tour_place$Country[k] <- "Great Britain"
  }
}

# adding country codes
tour_place$Country_code <- countrycode(tour_place$Country, origin = "country.name", destination = "iso3c")

# adding winner home advantage column
data_matches_no_Davis_1991 <- merge(data_matches_no_Davis_1991, tour_place, by.x = "tourney_name", by.y = "City", all.x = TRUE)
data_matches_no_Davis_1991$winner_home_advantage <- ifelse(data_matches_no_Davis_1991$winner_ioc == data_matches_no_Davis_1991$Country_code, 1, 0)

#converting to dummy variable (factor)
data_matches_no_Davis_1991$winner_home_advantage <- as.factor(data_matches_no_Davis_1991$winner_home_advantage)

summary(data_matches_no_Davis_1991)

# adding loser home (disadvantage?) column
data_matches_no_Davis_1991$loser_home <- ifelse(data_matches_no_Davis_1991$loser_ioc == data_matches_no_Davis_1991$Country_code, 1, 0)
data_matches_no_Davis_1991$loser_home <- as.factor(data_matches_no_Davis_1991$loser_home)


# ------------------------- Surface Dummies --------------------------------
data_matches$hard <- ifelse(data_matches$surface == "Hard", 1, 0)
data_matches$hard <- as.factor(data_matches$hard)

data_matches$carpet <- ifelse(data_matches$surface == "Carpet", 1, 0)
data_matches$carpet <- as.factor(data_matches$carpet)

data_matches$clay <- ifelse(data_matches$surface == "Clay", 1, 0)
data_matches$clay <- as.factor(data_matches$clay)

data_matches$grass <- ifelse(data_matches$surface == "Grass", 1, 0)
data_matches$grass <- as.factor(data_matches$grass)

summary(data_matches)
#the 1's don't add up

sum(data_matches$surface == "Hard") # 72 752
sum(data_matches$surface == "Carpet") # 20 355
sum(data_matches$surface == "Clay") # 65 671
sum(data_matches$surface == "Grass") # 22 212

sum(data_matches$surface != "Hard" & data_matches$surface != "Carpet" &
      data_matches$surface != "Clay" & data_matches$surface != "Grass") # 2 280

which(data_matches$surface != "Hard" & data_matches$surface != "Carpet" &
        data_matches$surface != "Clay" & data_matches$surface != "Grass")

#several matches have empty surface info -> assigning NA (these will be deleted later)
for(j in 1:length(data_matches$surface)){
  if(data_matches$surface[j] != "Hard" & data_matches$surface[j] != "Carpet" &
     data_matches$surface[j] != "Clay" & data_matches$surface[j] != "Grass"){
    data_matches$surface[j] <- NA
  }
}

# ---------------------------- Tournament Year -------------------------------

years <- sapply(data_matches$tourney_date, function(x) substr(x, 1, 4))
data_matches$tournament_year <- years


# ----------------------------- Head-to-Head ---------------------------------

#calculating winner's H2H value
# function to calculate cumulative head-to-head (h2h) count for each match
winner_h2h <- function(data_matches) {
  cumulative_h2h <- list()
  for (i in 1:nrow(data_matches)) {
    match <- data_matches[i, ]
    winner <- match$winner_id
    loser <- match$loser_id
    # getting all previous matches between the winner and loser
    previous_matches <- data_matches %>%
      filter((winner_id == winner & loser_id == loser) | (winner_id == loser & loser_id == winner)) %>%
      filter(tourney_date < match$tourney_date)
    # calculating the cumulative h2h count
    cumulative_h2h[i] <- sum(previous_matches$winner_id == winner)
  }
  return(unlist(cumulative_h2h))
}

data_matches$winner_h2h <- winner_h2h(data_matches)


#calculating loser's H2H value
loser_h2h <- function(data_matches) {
  cumulative_h2h <- list()
  for (i in 1:nrow(data_matches)) {
    match <- data_matches[i, ]
    winner <- match$winner_id
    loser <- match$loser_id
    # getting all previous matches between the winner and loser (from loser's perspective)
    previous_matches <- data_matches %>%
      filter((winner_id == loser & loser_id == winner) | (winner_id == winner & loser_id == loser)) %>%
      filter(tourney_date < match$tourney_date)
    # calculating the cumulative h2h count
    cumulative_h2h[i] <- sum(previous_matches$winner_id == loser)
  }
  return(unlist(cumulative_h2h))
}

data_matches$loser_h2h <- loser_h2h(data_matches)


# ---------------------------- Match Statistics -------------------------------
avg_player_stats <- function(data, winner_col, loser_col, winner_specific_col, loser_specific_col) {
  data %>%
    group_by(player_id = {{ winner_col }}) %>%
    mutate(
      w_bpFaced_avg = lag(cummean({{ winner_specific_col }}))
    ) %>%
    ungroup() %>%
    group_by(player_id = {{ loser_col }}) %>%
    mutate(
      l_bpFaced_avg = lag(cummean({{ loser_specific_col }}))
    ) %>%
    ungroup() %>%
    dplyr::select(-player_id) 
}

#selecting dataset with only non-NA entries in the match-stat variables
data_matches_stat_no_na <- data_matches[complete.cases(data_matches[, 28:45]), ]

average_data <- avg_player_stats(data_matches_stat_no_na, winner_id, loser_id, w_ace, l_ace)
average_data <- avg_player_stats(average_data, winner_id, loser_id, w_df, l_df)

data_match_stats <- average_data[, c(50, 101:118)]
data_matches <- merge(data_matches, data_match_stats, by = "row_number", all.x = TRUE)


# ----------------------- Player's Hand Dummies -----------------------
#winner
data_matches$winner_hand_R <- ifelse(data_matches$winner_hand == "R", 1, 0)
data_matches$winner_hand_R <- as.factor(data_matches$winner_hand_R)

data_matches$winner_hand_L <- ifelse(data_matches$winner_hand == "L", 1, 0)
data_matches$winner_hand_L <- as.factor(data_matches$winner_hand_L)

#loser
data_matches$loser_hand_R <- ifelse(data_matches$loser_hand == "R", 1, 0)
data_matches$loser_hand_R <- as.factor(data_matches$loser_hand_R)

data_matches$loser_hand_L <- ifelse(data_matches$loser_hand == "L", 1, 0)
data_matches$loser_hand_L <- as.factor(data_matches$loser_hand_L)

summary(data_matches$winner_hand_R)

#some hand entries are empty ("") or undefined ("U")
sum(data_matches$winner_hand == "U") #1853
sum(data_matches$winner_hand == "") #17
sum(data_matches$loser_hand == "U") #5734
sum(data_matches$loser_hand == "") #63

#replacing them by NA (to be deleted later)
data_matches$winner_hand <- ifelse(data_matches$winner_hand == "U" | 
                                     data_matches$winner_hand == "", NA, data_matches$winner_hand)

data_matches$loser_hand <- ifelse(data_matches$loser_hand == "U" | 
                                    data_matches$loser_hand == "", NA, data_matches$loser_hand)



#write.table(data_matches, paste0(getwd(), "/data_matches_final.csv"), sep=";")

data_matches_final <- read.csv(paste0(getwd(), "/data_matches_final.csv"),
                               header = TRUE,
                               sep = ";",
                               dec = ".")


####################### Putting Together Datasets ##########################

data_matches_final_1991 <- data_matches_final[data_matches_final$tourney_date >= "1991-01-01", ]
data_matches_final_no_Davis_1991 <- data_matches_final_1991[data_matches_final_1991$tourney_level != "D", ] # 89993

#merging data_matches_final_no_Davis_1991 & data_matches_no_Davis_1991 in correct order
home_advantage_df <- data_matches_no_Davis_1991[, c(50:54)]
row_order <- data_matches_final_no_Davis_1991$row_number

home_advantage_df <- home_advantage_df[match(row_order, home_advantage_df$row_number), ]

data_matches_final_no_Davis_1991$tourney_country_code <- home_advantage_df$Country_code
data_matches_final_no_Davis_1991$winner_home <- home_advantage_df$winner_home_advantage
data_matches_final_no_Davis_1991$loser_home <- home_advantage_df$loser_home

summary(data_matches_final_no_Davis_1991) # 89 993 observations


###################### Deleting Missing Values ######################
#dataset1: keep winner_seed & loser_seed -> resulting in a very small dataset

# delete missing winner_ht, loser_ht, loser_age, minutes, 
# w_ace - loser_rank_points, RR- Final, 
# rows where player's hand is either U or nothing

# --- dataset with no NA values, including columns winner_seed & looser_seed columns
#-> 7810 observations
data1_1991 <- data_matches_final_no_Davis_1991[complete.cases(data_matches_final_no_Davis_1991), ]

unique(data1_1991$tourney_level) #-> doesn't include any Davis cup or Tour Final matches

#removing dummy columns for Davis cup and Tour Finals
data1_1991 <- data1_1991[, -which(names(data1_1991) %in% c("Tour_Finals", "Davis_Cup"))]

unique(data1_1991$round) #-> doesn't include any RR or R128 matches

#removing RR and R128 dummy columns
data1_1991 <- data1_1991[, -which(names(data1_1991) %in% c("RR", "R128"))]


############################# Target Variable #################################

# function for extracting last name
extract_last_name <- function(full_name) {
  name_parts <- strsplit(full_name, " ")
  last_name <- sapply(name_parts, function(x) x[length(x)])
  return(last_name)
}

# adding columns with last names
data1_1991 <- data1_1991 %>%
  mutate(winner_last_name = extract_last_name(winner_name),
         loser_last_name = extract_last_name(loser_name))

# determining which last name comes first alphabetically
data1_1991 <- data1_1991 %>%
  mutate(FirstAlphabetically = ifelse(winner_last_name < loser_last_name, winner_last_name, loser_last_name))

data1_1991$target <- rep(NA, nrow(data1_1991))

#removing unnecessary columns
reg_data1_1991 <- data1_1991[, c(5, 10, 12, 14, 16, 18, 20, 22, 24, 26, 47:74, 76:114, 116:121)]
reg_data1_1991$winner_home <- as.numeric(as.character(reg_data1_1991$winner_home))
reg_data1_1991$loser_home <- as.numeric(as.character(reg_data1_1991$loser_home))


pairs_to_swap <- list(c("winner_name", "loser_name"), c("winner_seed", "loser_seed"), 
                      c("winner_ht", "loser_ht"), c("winner_age", "loser_age"), 
                      c("w_ace_avg", "l_ace_avg"), c("w_df_avg", "l_df_avg"), 
                      c("w_svpt_avg", "l_svpt_avg"), 
                      c("w_1stIn_avg", "l_1stIn_avg"), 
                      c("w_1stWon_avg", "l_1stWon_avg"), 
                      c("w_2ndWon_avg", "l_2ndWon_avg"), 
                      c("w_SvGms_avg", "l_SvGms_avg"), 
                      c("w_bpSaved_avg", "l_bpSaved_avg"), 
                      c("w_bpFaced_avg", "l_bpFaced_avg"), 
                      c("winner_rank", "loser_rank"), 
                      c("winner_rank_points", "loser_rank_points"), 
                      c("winner_abs_wins_clay", "loser_abs_wins_clay"), 
                      c("winner_abs_wins_hard", "loser_abs_wins_hard"), 
                      c("winner_abs_wins_carpet", "loser_abs_wins_carpet"), 
                      c("winner_abs_wins_grass", "loser_abs_wins_grass"), 
                      c("loser_abs_losses_clay", "winner_abs_losses_clay"), 
                      c("loser_abs_losses_hard", "winner_abs_losses_hard"), 
                      c("loser_abs_losses_carpet", "winner_abs_losses_carpet"), 
                      c("loser_abs_losses_grass", "winner_abs_losses_grass"), 
                      c("winner_abs_wins_level1", "loser_abs_wins_level1"),
                      c("winner_abs_wins_level2", "loser_abs_wins_level2"),
                      c("loser_abs_losses_level1", "winner_abs_losses_level1"), 
                      c("loser_abs_losses_level2", "winner_abs_losses_level2"), 
                      c("winner_h2h", "loser_h2h"), 
                      c("winner_hand_R", "loser_hand_R"), 
                      c("winner_hand_L", "loser_hand_L"), 
                      c("winner_home", "loser_home"))


for (i in 1:nrow(reg_data1_1991)) {
  w_last_name <- extract_last_name(reg_data1_1991$winner_name[i])
  first_alpha_last_name <- reg_data1_1991$FirstAlphabetically[i]
  
  if (w_last_name != first_alpha_last_name) {
    for (pair in pairs_to_swap) {
      col1 <- pair[1]
      col2 <- pair[2]
      
      temp <- reg_data1_1991[i, col1]
      reg_data1_1991[i, col1] <- reg_data1_1991[i, col2]
      reg_data1_1991[i, col2] <- temp
    }
  }
}

reg_data1_1991$target <- ifelse(reg_data1_1991$FirstAlphabetically == reg_data1_1991$winner_last_name, 1, 0)

#renaming columns
tennis_col_names <- colnames(reg_data1_1991)
new_col_names <- gsub("winner", "p1", tennis_col_names)
new_col_names <- gsub("loser", "p2", new_col_names)
new_col_names[60:77] <- c("p1_ace_avg", "p2_ace_avg", "p1_df_avg", "p2_df_avg", "p1_svpt_avg",
                          "p2_svpt_avg", "p1_1stIn_avg", "p2_1stIn_avg", "p1_1stWon_avg" ,"p2_1stWon_avg",
                          "p1_2ndWon_avg", "p2_2ndWon_avg", "p1_SvGms_avg", "p2_SvGms_avg", "p1_bpSaved_avg",
                          "p2_bpSaved_avg", "p1_bpFaced_avg", "p2_bpFaced_avg")
new_col_names[80:81] <- c("winner_last_name", "loser_last_name")

colnames(reg_data1_1991) <- new_col_names

#write.table(reg_data1_1991, paste0(getwd(), "/reg_data1_1991NEW.csv"), sep=";")


############################# MACHINE LEARNING PART ###########################

# all of the code above is selected from the code I used for Big Data course
# some of the dataset dimensions do not match as I used more features in Big Data
# which I have not included here

# for simplicity and time efficiency, I load the final dataset from Big Data
# instead of running everything again
reg_data1_1991 <- read.csv(paste0(getwd(), "/reg_data1_1991NEW.csv"),
                           header = TRUE,
                           sep = ";",
                           dec = ".")


# transforming some of the explanatory variables
reg_data1_1991$age_diff <- reg_data1_1991$p1_age - reg_data1_1991$p2_age
reg_data1_1991$seed_diff <- reg_data1_1991$p1_seed - reg_data1_1991$p2_seed
reg_data1_1991$rank_diff <- reg_data1_1991$p1_rank - reg_data1_1991$p2_rank
reg_data1_1991$ranking_points_diff <- reg_data1_1991$p1_rank_points - reg_data1_1991$p2_rank_points
reg_data1_1991$height_diff <- reg_data1_1991$p1_ht - reg_data1_1991$p2_ht

X <- reg_data1_1991[, !colnames(reg_data1_1991) %in%
                      c("p1_name", "p2_name", "carpet",
                        "Another_level", "R64",
                        "p1_hand_L", "p2_hand_L",
                        "winner_last_name",
                        "loser_last_name",
                        "FirstAlphabetically",
                        "p1_age", "p2_age", "p1_seed",
                        "p2_seed", "p1_rank", "p2_rank",
                        "p1_rank_points", "p2_rank_points",
                        "p1_ht", "p2_ht", "target")][,-1]

X <- X[, -c(5:31, 35:39, 46:59)]

y <- reg_data1_1991$target
y <- factor(y, levels = c(0, 1))

tennis_data <- cbind(X, y)


# --- splitting data into training and test set:
# training set: 70% -> 5467 observations
# test set: 30% of the data -> 2343 observations

n <- nrow(X)
set.seed(123)
draws_train <- sample.int(n, 0.7*n)
train <- 1:n %in% draws_train
test <- !train



############################ Decision Tree ##################################

# fitting classification tree:

# using cross-entropy
tree_model_dev <- tree(y ~., data = tennis_data[train,], split = "deviance") 
summary(tree_model_dev)
plot(tree_model_dev)
text(tree_model_dev, pretty = 0, cex = 0.8)
tree_model_dev

dev_pred <- predict(tree_model_dev, tennis_data[test,], type = "class")
sum(dev_pred != y[test])

# using gini-index
tree_model_gini <- tree(y ~., data = tennis_data[train,], split = "gini") 
summary(tree_model_gini)
plot(tree_model_gini)
text(tree_model_gini, pretty = 0, cex = 0.8)
tree_model_gini

gini_pred <- predict(tree_model_gini, tennis_data[test,], type = "class")
sum(gini_pred != y[test])

# pruning the tree -> finding optimal level of complexity

# for model using cross-entropy:
set.seed(3)
cv_dev <- cv.tree(tree_model_dev, FUN = prune.misclass)
cv_dev

#par(mfrow = c(1,2))
plot(cv_dev$size, cv_dev$dev, type = "b", xlab = "size", ylab = "misclassification error")
#plot(cv_dev$k, cv_dev$dev, type = "b", xlab = "k", ylab = "deviance")
par(mfrow = c(1,1))

prune_dev <- prune.misclass(tree_model_dev, best = 8)
plot(prune_dev)
text(prune_dev, pretty = 0, cex = 0.8)

tree_dev_pred <- predict(prune_dev, tennis_data[test,], type = "class")
table(tree_dev_pred, y[test])
sum(tree_dev_pred != y[test])


# for model using gini-index:
cv_gini <- cv.tree(tree_model_gini, FUN = prune.misclass)
cv_gini

#par(mfrow = c(1,2))
plot(cv_gini$size, cv_gini$dev, type = "b", xlab = "size", ylab = "misclassification error")
#plot(cv_gini$k, cv_gini$dev, type = "b")
#par(mfrow = c(1,1))

prune_gini <- prune.misclass(tree_model_gini, best = 100)
plot(prune_gini)
text(prune_gini, pretty = 0, cex = 0.6)

tree_gini_pred <- predict(prune_gini, tennis_data[test,], type = "class")
table(tree_gini_pred, y[test])
sum(tree_gini_pred != y[test])



################################ Boosting ####################################

# ---------------------------- Tuning Hyperparameters -------------------------

# boosting tuning grids:
boost_grid <- expand.grid(n.trees = c(400, 800, 1500),
                          interaction.depth = c(1, 2, 4, 8),
                          n.minobsinnode = c(1, 4, 8),
                          shrinkage = c(0.01))

levels(tennis_data$y) <- make.names(levels(tennis_data$y))

boost_control_method <- trainControl(method = "cv", number = 5,
                                        classProbs = TRUE, verboseIter = TRUE,
                                        #summaryFunction = twoClassSummary,
                                        savePredictions = "final")

boost_tuning <- caret::train(y ~ ., data = tennis_data[train,],
                            method = "gbm",
                            trControl = boost_control_method,
                            tuneGrid = boost_grid,
                            metric = "Accuracy",
                            verbose = TRUE,
                            distribution = "bernoulli")

# ---------------------- Predictions Using the Best Model ---------------------

boost_pred_probs <- predict(boost_tuning, newdata = tennis_data[test,], type = "prob")
boost_pred <- 1*(boost_pred_probs[,2] > 0.5)
sum(boost_pred != y[test])

relative.influence(boost_tuning$finalModel)

# ------------------ Boost Evaluation Using Different Measures ----------------

table(boost_pred, y[test])

#accuracy:
boost_acc <- sum(boost_pred == y[test])/length(y[test])

#precision:
boost_precision <- table(boost_pred, y[test])[2,2]/(table(boost_pred, y[test])[2,1] + table(boost_pred, y[test])[2,2])

#recall:
boost_recall <- table(boost_pred, y[test])[2,2]/(table(boost_pred, y[test])[1,2] + table(boost_pred, y[test])[2,2])

#F1 score:
boost_f1 <- 2*(boost_precision*boost_recall)/(boost_precision + boost_recall)

#ROC & AUC:
pos_class_probs_boost <- boost_pred_probs$X1
roc_obj_boost <- roc(y[test], pos_class_probs_boost)
auc_value_boost <- auc(roc_obj_boost)

plot.roc(roc_obj_boost, col = "#1c61b6", main = paste("Boosting ROC Curve (AUC =", round(auc_value_boost, 2), ")"))

boost_acc
boost_precision
boost_recall
boost_f1



############################### Bagging #####################################

# -> using all explanatory variables in each training

# ------------------------- Hyperparameter Tuning ----------------------------
tennis_data$y <- as.factor(tennis_data$y)


bagg_grid2 <- expand.grid(ntree = c(800, 1000, 3000, 5000),
                         sampsize = c(500, 800, 1200, 2000),
                         nodesize = c(1, 3, 5, 10),
                         maxnodes = c(8, 15, 30, 500))

bagg_grid2$mean.oob.error <- rep(NA, nrow(bagg_grid2))


for(r in 193:nrow(bagg_grid2)){
  print(r)
  bagg_tuning <- randomForest(y ~., data = tennis_data[train,],
                      ntree = bagg_grid2$ntree[r], 
                     sampsize = bagg_grid2$sampsize[r],
                     nodesize = bagg_grid2$nodesize[r],
                     maxnodes = bagg_grid2$maxnodes[r],
                     mrty = ncol(tennis_data)-1, 
                     replace = TRUE, 
                     importance = TRUE)
  
  bagg_grid2$mean.oob.error[r] <- mean(bagg_tuning$err.rate[, "OOB"])
}

# adding accuracy measure
bagg_grid2$accuracy <- 1 - bagg_grid2$mean.oob.error

which.max(bagg_grid2$accuracy)


# ---------------------- Predictions Using the Best Model ---------------------
bagg_best <- randomForest(y ~., data = tennis_data[train,],
                         ntree = 5000,
                         sampsize = 2000,
                         nodesize = 3,
                         maxnodes = 500,
                         mrty = ncol(tennis_data)-1,
                         replace = TRUE,
                         importance = TRUE)



bagg_pred <- predict(bagg_best, newdata = tennis_data[test,], type = "class")
bagg_pred_probs <- predict(bagg_best, newdata = tennis_data[test,], type = "prob")[,2]
sum(bagg_pred != y[test])

# assessing feature importance
importance_bagg <- importance(bagg_best, type = 1)

importance_df_bagg <- data.frame(
  Feature = row.names(importance_bagg),
  Importance = importance_bagg[, 1]
)

ggplot(importance_df_bagg, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "red") +
  geom_errorbar(aes(ymin = Importance - sd(Importance), ymax = Importance + sd(Importance)), width = 0.4) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importances using Bagging Model",
       x = "Features",
       y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# -------- Bagging Partial Dependence Plots for Important Variables -----------

# PDP using ace_avg
partialPlot(bagg_best, tennis_data[train,], 
            x.var = "p1_ace_avg", 
            main = "Partial Dependence Plot on p1_ace_avg", 
            xlab = "p1_ace_avg")

partialPlot(bagg_best, tennis_data[train,], 
            x.var = "p2_ace_avg", 
            main = "Partial Dependence Plot on p2_ace_avg", 
            xlab = "p2_ace_avg")

# PDP using df_avg
partialPlot(bagg_best, tennis_data[train,], 
            x.var = "p1_df_avg", 
            main = "Partial Dependence Plot on p1_df_avg", 
            xlab = "p1_df_avg")

partialPlot(bagg_best, tennis_data[train,], 
            x.var = "p2_df_avg", 
            main = "Partial Dependence Plot on p2_df_avg", 
            xlab = "p2_df_avg")

# PDP using tournament_year
partialPlot(bagg_best, tennis_data[train,], 
            x.var = "tournament_year", 
            main = "Partial Dependence Plot on tournament_year", 
            xlab = "tournament_year")


# ------------------ Bagging Evaluation Using Different Measures -----------------

table(bagg_pred, y[test])

#accuracy:
bagg_acc <- sum(bagg_pred == y[test])/length(y[test])

#precision:
bagg_precision <- table(bagg_pred, y[test])[2,2]/(table(bagg_pred, y[test])[2,1] + table(bagg_pred, y[test])[2,2])

#recall:
bagg_recall <- table(bagg_pred, y[test])[2,2]/(table(bagg_pred, y[test])[1,2] + table(bagg_pred, y[test])[2,2])

#F1 score:
bagg_f1 <- 2*(bagg_precision*bagg_recall)/(bagg_precision + bagg_recall)

#ROC & AUC:

roc_obj_bagg <- roc(y[test], bagg_pred_probs)
auc_value_bagg <- auc(roc_obj_bagg)

plot.roc(roc_obj_bagg, col = "#1c61b6", main = paste("Bagging ROC Curve (AUC =", round(auc_value_bagg, 2), ")"))

bagg_acc
bagg_precision
bagg_recall
bagg_f1



############################# Random Forests #################################

# -> improvement over bagging by interacting only a certain amount of variables

# ------------------------- Hyperparameter Tuning -----------------------------

rf_grid <- expand.grid(ntree = c(800, 1000, 3000, 5000),
                            sampsize = c(500, 800, 1200, 2000),
                            nodesize = c(1, 3, 5, 10),
                            maxnodes = c(8, 15, 30, 500),
                            mtry = c(2, 5, 7, 10))


rf_grid$mean.oob.error <- rep(NA, nrow(rf_grid))

tennis_data$y <- as.factor(tennis_data$y)


for(r in 1:nrow(rf_grid)){
  print(r)
  rf_tuning <- randomForest(y ~., data = tennis_data[train,],
                              ntree = rf_grid$ntree[r], 
                              sampsize = rf_grid$sampsize[r],
                              nodesize = rf_grid$nodesize[r],
                              maxnodes = rf_grid$maxnodes[r],
                              mrty = rf_grid$mrty[r], replace = TRUE, 
                              importance = TRUE)
  
  rf_grid$mean.oob.error[r] <- mean(rf_tuning$err.rate[, "OOB"])
}

# adding accuracy measure
rf_grid$accuracy <- 1 - rf_grid$mean.oob.error

which.max(rf_grid$accuracy)


# --------------------- RF Predictions Using the Best Model -------------------

rf_best <- randomForest(y ~., data = tennis_data[train,],
                        ntree = 5000,
                        sampsize = 2000,
                        nodesize = 1,
                        maxnodes = 500,
                        mrty = 10,
                        replace = TRUE,
                        importance = TRUE)

rf_pred <- predict(rf_best, newdata = tennis_data[test,], type = "class")
rf_pred_probs <- predict(rf_best, newdata = tennis_data[test,], type = "prob")[, 2]
sum(rf_pred != y[test])

# assessing feature importance
importance_rf <- importance(rf_best, type = 1)

importance_df_rf <- data.frame(
  Feature = row.names(importance_rf),
  Importance = importance_rf[, 1]
)

ggplot(importance_df_rf, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "red") +
  geom_errorbar(aes(ymin = Importance - sd(Importance), ymax = Importance + sd(Importance)), width = 0.4) +
  coord_flip() +
  theme_minimal() +
  labs(title = "Feature Importances using Random Forest",
       x = "Features",
       y = "Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ------------------- RF Evaluation Using Different Measures ------------------

table(rf_pred, y[test])

#accuracy:
rf_acc <- sum(rf_pred == y[test])/length(y[test])

#precision:
rf_precision <- table(rf_pred, y[test])[2,2]/(table(rf_pred, y[test])[2,1] + table(rf_pred, y[test])[2,2])

#recall:
rf_recall <- table(rf_pred, y[test])[2,2]/(table(rf_pred, y[test])[1,2] + table(rf_pred, y[test])[2,2])

#F1 score:
rf_f1 <- 2*(rf_precision*rf_recall)/(rf_precision + rf_recall)

#ROC & AUC:

roc_obj_rf <- roc(y[test], rf_pred_probs)
auc_value_rf <- auc(roc_obj_rf)

plot.roc(roc_obj_rf, col = "#1c61b6", main = paste("Random Forrest ROC Curve (AUC =", round(auc_value_rf, 2), ")"))

rf_acc
rf_precision
rf_recall
rf_f1



############################# Calibration Curves ##############################

# -------------------- Calibration Curve for Boosting Model --------------------

#colnames(boost_pred_probs) <- c(0,1)

# calculating calibration curve
calibrationData_boost <- data.frame(observed = tennis_data$y[test], predicted = boost_pred_probs[,2])
#colnames(calibrationData_boost) <- c("observed", 0, 1)
calibrationData_boost$bin <- cut(calibrationData_boost$predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)

calibrationData_boost$observed <- as.numeric(as.character(calibrationData_boost$observed))

calibrationStats_boost <- calibrationData_boost %>%
  group_by(bin) %>%
  summarise(mean_predicted = mean(predicted), 
            fraction_positive = mean(observed))


# plotting calibration curve
ggplot(calibrationStats_boost, aes(x = mean_predicted, y = fraction_positive)) +
  geom_point() +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(x = "Mean predicted value", y = "Fraction of positives") +
  ggtitle("Calibration Curve for Boosting Model")


# -------------------- Calibration Curve for Bagging Model --------------------

# calculating calibration curve
calibrationData_bagg <- data.frame(observed = tennis_data$y[test], predicted = bagg_pred_probs)
calibrationData_bagg$bin <- cut(calibrationData_bagg$predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)

calibrationData_bagg$observed <- as.numeric(as.character(calibrationData_bagg$observed))

calibrationStats_bagg <- calibrationData_bagg %>%
  group_by(bin) %>%
  summarise(mean_predicted = mean(predicted), 
            fraction_positive = mean(observed))


# plotting calibration curve
ggplot(calibrationStats_bagg, aes(x = mean_predicted, y = fraction_positive)) +
  geom_point() +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(x = "Mean predicted value", y = "Fraction of positives") +
  ggtitle("Calibration Curve for Bagging Model")


# ---------------------- Calibration Curve for RF Model -----------------------

# calculating calibration curve
calibrationData_rf <- data.frame(observed = tennis_data$y[test], predicted = rf_pred_probs)
calibrationData_rf$bin <- cut(calibrationData_rf$predicted, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)

calibrationData_rf$observed <- as.numeric(as.character(calibrationData_rf$observed))

calibrationStats_rf <- calibrationData %>%
  group_by(bin) %>%
  summarise(mean_predicted = mean(predicted), 
            fraction_positive = mean(observed))


# plotting calibration curve
ggplot(calibrationStats_rf, aes(x = mean_predicted, y = fraction_positive)) +
  geom_point() +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(x = "Mean predicted value", y = "Fraction of positives") +
  ggtitle("Calibration Curve for Random Forest Model")



############################# Shapley Value ##################################

set.seed(2)
draws_ind <- sample(draws_train, 500)
ind <- 1:n %in% draws_ind
print(ind)


model_rf <- randomForest(y ~., data = tennis_data[train,],
                         ntree = 800, 
                         sampsize = 1000,
                         nodesize = 5,
                         maxnodes = 100,
                         mrty = 15, replace = TRUE, 
                         importance = TRUE,
                         type = "classification")

p_function <- function(model, data) predict(model, newdata = data, type = "prob")

s <- kernelshap::kernelshap(model_rf, tennis_data[train, -which(names(tennis_data) == "y")], 
                            bg_X = tennis_data[ind, -which(names(tennis_data) == "y")], 
                            pred_fun = p_function)
sv <- shapviz::shapviz (s)
shapviz::sv_importance(sv)


