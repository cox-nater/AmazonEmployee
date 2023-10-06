library(tidymodels)
library(embed)

amazon_train <- vroom('C:/BYU/2023(5) Fall/STAT 348/AmazonEmployee/train.csv',
                    show_col_types = FALSE) %>%
  select(2:10,1)

ggplot(amazon_train)+
  geom_point(mapping =aes(x=ROLE_CODE, y=ACTION))
ggplot(amazon_train)+
  geom_point(mapping =aes(x=ROLE_ROLLUP_1, y=ACTION))
ggplot(amazon_train)+
  geom_point(mapping =aes(x=ROLE_ROLLUP_2, y=ACTION))

my_recipe <- recipe(ACTION~., data = amazon_train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .01) %>%
  step_dummy(all_nominal_predictors())%>%
  step_lencode_mixed(all_nominal_predictors(), outcome= vars(target_var))
  # also step_lencode_glm() and step_lencode_bayes()

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data= amazon_train)
