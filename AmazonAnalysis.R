library(tidymodels)
library(embed)
library(vroom)
library(tidyverse)
library(doParallel)

cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

amazon_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/AmazonEmployee/train.csv',
                    show_col_types = FALSE) %>%
  select(2:10,1) %>%
  mutate(ACTION = as.factor(ACTION))

amazon_test <- vroom('C:/BYU/2023(5) Fall/STAT 348/AmazonEmployee/test.csv',
                     show_col_types = FALSE)

#ggplot(amazon_train)+
#  geom_point(mapping =aes(x=ROLE_CODE, y=ACTION))
#ggplot(amazon_train)+
#  geom_point(mapping =aes(x=ROLE_ROLLUP_1, y=ACTION))
#ggplot(amazon_train)+
#  geom_point(mapping =aes(x=ROLE_ROLLUP_2, y=ACTION))

my_recipe <- recipe(ACTION~., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  #step_dummy(all_nominal_predictors())%>%
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(ACTION))
  # also step_lencode_glm() and step_lencode_bayes()

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data= amazon_train)

logRegModel <- logistic_reg(mixture = tune() , penalty = tune() ) %>%
  set_engine("glmnet")

logReg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel)


tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

CV_results <- logReg_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc)) #, f_meas, sens, recall, spec, precision, accuracy))

bestTune <- CV_results %>%
  select_best("roc_auc")

final_logReg_wf <- logReg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

amazon_predictions <- predict(final_logReg_wf,
                              new_data = amazon_test,
                              type= "prob") %>%
  # bind_cols(amazon_test)%>%
  mutate(Action = .pred_1, #ifelse(.pred_1 >= .96, 1, 0),
         ID = id) %>%
  select(ID, Action)
stopCluster(cl)

vroom_write(x=amazon_predictions, file="./TestPreds.csv", delim=",")
#save(file=".AmazonAnalysis.R", list=c())

