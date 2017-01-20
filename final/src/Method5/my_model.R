## Read all data frames produced by 'my_features.py'
## Remove words with any NA in its features
## (e.g., if a word doesn't appear in any title or in any content, we should not consider it.)
bio.df <- read.csv("./biology_df.csv", stringsAsFactors = F)
bio.df <- bio.df[complete.cases(bio.df),]

cook.df <- read.csv("./cooking_df.csv", stringsAsFactors = F)
cook.df <- cook.df[complete.cases(cook.df),]

cryp.df <- read.csv("./crypto_df.csv", stringsAsFactors = F)
cryp.df <- cryp.df[complete.cases(cryp.df),]

diy.df <- read.csv("./diy_df.csv", stringsAsFactors = F)
diy.df <- diy.df[complete.cases(diy.df),]

rob.df <- read.csv("./robotics_df.csv", stringsAsFactors = F)
rob.df <- rob.df[complete.cases(rob.df),]

trv.df <- read.csv("./travel_df.csv", stringsAsFactors = F)
trv.df <- trv.df[complete.cases(trv.df),]

phy.df <- read.csv("./test_df.csv", stringsAsFactors = F)
phy.df <- phy.df[complete.cases(phy.df),]

## Use 'robotics_df.csv' as the testing data and all the rests as the training data.
testing <- rob.df
## Don't use 'travel_df.csv', it sucks.
training <- rbind(bio.df, cook.df, cryp.df, diy.df)
## Convert outcome (is_tag) as factor for later randomForest() function
training$is_tag <- factor(ifelse(training$is_tag == 0, "NO", "YES"), levels = c("NO", "YES"))
testing$is_tag <- factor(ifelse(testing$is_tag == 0, "NO", "YES"), levels = c("NO", "YES"))
## Specify the formula
formula <- reformulate(setdiff(names(training), c("word", "is_tag")), response = "is_tag")

## Build random forest model
library(randomForest)
set.seed(1002)
## Run 10 iterations of balancing to check the average performance
for (i in 1:10) {
    ## Balancing
    r0 <- training[training$is_tag == "NO",]
    r0 <- r0[sample(1:nrow(r0), size = sum(training$is_tag == "YES")),]
    training.bal <- rbind(r0, training[training$is_tag == "YES",])
    ## Build random forest model
    rf.fit <- randomForest(formula, data = training.bal)
    ## Validation
    pred.prob <- predict(rf.fit, newdata = testing, type = "prob")
    pred.value <- predict(rf.fit, newdata = testing)
    accuracy <- sum(pred.value == testing$is_tag) / nrow(testing)
    print(accuracy)
}

## Build a random forest model again and predict the probability
## of being a valid tag for all words in physics questions
rf.fit <- randomForest(formula, data = training.bal, importance = T)
pred.prob <- predict(rf.fit, newdata = phy.df, type = "prob")
tag_prob <- data.frame(word = phy.df$word,
                       prob = pred.prob[,2],
                       noun_ratio = phy.df$ratio_noun)
## Write the result
write.csv(tag_prob, "./physics_prob.csv", row.names = F)
## Check the feature importance:
print(importance(rf.fit)[,3])

## TODO:
### Does number of words in a question relate to number of tags?
### Model for n-grams?

## -----------------------------------------
## backups: Use R code to generate features?
## -----------------------------------------
# biology <- read.csv("~/Documents/NTU/ML2016/program/ML2016/final_tags/other/biology.csv",
#                     header = T,
#                     stringsAsFactors = F)
# dim(biology)
# biology.n.tags <- sapply(strsplit(biology$tags, split = " "), length)
# mean(biology.n.tags)
# max(biology.n.tags)
# all.tags.biology <- unlist(strsplit(biology$tags, split = " "))
# length(unique(all.tags.biology))
# # head(sort(table(all.tags.biology)), 20)
# # tail(sort(table(all.tags.biology)), 20)
# 
# unique.tag.biology <- unique(all.tags.biology)
# unique.ngram_tag.biology <- data.frame(tag = unique.tag.biology[grepl("-", unique.tag.biology)],
#                                        stringsAsFactors = F)
# unique.ngram_tag.biology$n <- sapply(unique.ngram_tag.biology$tag, function(str) length(strsplit(str, "-")[[1]]))
# sum(unique.ngram_tag.biology$n == 2)
# sum(unique.ngram_tag.biology$n == 3)
# 
# i <- 11
# ## Preprocessing
# title <- gsub("[^[:alnum:][:space:]-]", "", biology$title[i])
# content <- gsub("<.*?>", "", biology$content[i])
# content <- gsub("\n+", " ", content)
# content <- gsub("[^[:alnum:][:space:]-]", "", content)
# content <- gsub("^\\s+|\\s+$", "", content)
# title_content <- paste(title, content)
# ## Tokenization
# title_content.tokens <- strsplit(title_content, "\\s+")
# 
# tags <- biology$tags[i]
