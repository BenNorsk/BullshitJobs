# Load the libraries
library(did)
library(dplyr)

warnings()
options(warn = 1)


# Read in the data from data/master_data.csv
data <- read.csv("data/master_data_firm.csv")


# Print the number of observations
print(nrow(data))
print(head(data))


out <- att_gt(
    yname = "bs_score_llm",
    gname = "return_to_office_week_numeric",
    idname = "firm_id",
    tname = "date_week_numeric",
    xformla = ~1,
    data = data,
    est_method = "reg",
    panel = FALSE,
    control_group = "notyettreated"
)

# Print the summary of the output
summary(out)
