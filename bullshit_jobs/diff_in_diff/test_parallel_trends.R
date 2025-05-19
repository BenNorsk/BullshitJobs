library(did)
library(dplyr)

warnings()
options(warn = 1)

# Read in the data from data/master_data.csv
data <- read.csv("data/master_data_cross_section.csv")


# Print the number of observations
print(nrow(data))
print(head(data))

# Print all column names
print(colnames(data))


# Python Excerpt
# score_columns = [
#         'bs_score_binary_dict', 'bs_score_llm', 'rating',
#         'Career Opportunities', 'Compensation and Benefits',
#         'Senior Management', 'Work/Life Balance',
#         'Culture & Values', 'Diversity & Inclusion'
#     ]


# Define the covariates
covariates = c(
  "rating",
  "Career.Opportunities",
  "Senior.Management",
  "Culture...Values",
  "Compensation.and.Benefits",
  "Work.Life.Balance",
  "Diversity...Inclusion",
  "sector_technology",
  "sector_finance",
  "sector_manufacturing",
  "sector_telecom"
)

# Define the target variable
target_variable = "bs_score_llm"

# Define the other relevant variables
other_relevant_variables = c(
  "review_id", # I think I need that one...
  "firm_id", # For clustering
  "date_week_numeric", # For time
  "return_to_office_week_numeric" # For treatment
)

# Remove all the non relevant variables
data <- data %>%
  select(all_of(c(target_variable, other_relevant_variables, covariates))) %>%
  mutate(
    return_to_office_week_numeric = as.numeric(return_to_office_week_numeric),
    date_week_numeric = as.numeric(date_week_numeric)
  )

# Only keep at max. 1 observation per firm per week
data <- data %>%
  group_by(firm_id, date_week_numeric) %>%
  slice(1) %>%
  ungroup()

# Print the data
print(head(data))

# Print the number of observations
print(nrow(data))


# --- Conduct the Parallel Trends Test with Sector Covariates ---

pre.test <- conditional_did_pretest(
  yname = "bs_score_llm",  # Outcome variable
  tname = "date_week_numeric",  # Time variable
  idname = "review_id",  # Cross-sectional unit ID
  gname = "return_to_office_week_numeric",  # First treatment week
  xformla = ~ 1,  # Covariates (exclude telecom to avoid multicollinearity)
  data = data,
  panel = FALSE,
  allow_unbalanced_panel = FALSE,
  control_group = "notyettreated",
  weightsname = NULL,
  alp = 0.05,
  bstrap = TRUE,
  cband = TRUE,
  biters = 1000,
  clustervars = c("review_id", "firm_id"),  # Cluster at firm level
  est_method = "dr",
  print_details = TRUE,
  pl = FALSE,
  cores = 1
)

# View summary of results
summary(pre.test)

# Save output object to file for future use
saveRDS(pre.test, file = "data/results/pretest_output.rds")
