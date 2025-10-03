df <- read.csv('experiments_data/baseline_df.csv')

# Make sure Rater and Race are factors
df$Rater <- factor(df$Rater)
df$Race <- factor(df$Race)

library(lme4)
library(lmerTest)

# Fit the mixed effects model
model <- lmer(Prediction ~ Rater * Race + (1 | Patient.Number) + (1 | Rater), data = df)

# Display the summary of the model
summary(model)

# Assess variance components
VarCorr(model)

# ANOVA Testing

library(effectsize)
library(rstatix)

df <- read.csv('experiments_data/balanced_mean_diffs.csv')

# ANOVA
anova_result <- aov(Difference ~ Race, data = df)
summary(anova_result)
omega_squared(anova_result)

# Kruskal-Wallis test
kruskal_result <- kruskal_test(df, Difference ~ Race)
print(kruskal_result)

# Effect size (epsilon squared)
kruskal_effsize_result <- kruskal_effsize(df, Difference ~ Race)
print(kruskal_effsize_result)


