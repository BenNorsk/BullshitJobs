# Load the libraries
library(did)
library(dplyr)

# Convert date column to numeric

convert_dates_to_numeric <- function(df, date_cols) {
  for (col in date_cols) {
    # Convert to Date if not already
    df[[col]] <- as.Date(df[[col]])

    # Normalize: min = 0, max = max - min
    min_date <- min(df[[col]], na.rm = TRUE)
    df[[paste0(col, "_numeric")]] <- as.numeric(df[[col]] - min_date)
  }
  return(df)
}


# Read in the data from data/master_data.csv
data <- read.csv("data/master_data.csv")

# Convert the date columns to numeric
data <- convert_dates_to_numeric(data, c("date", "return_to_office"))


# Print the first few rows of the data
print(head(data))

# Print the max date_numeric
print(max(data$date_numeric, na.rm = TRUE))
# Print the min date_numeric
print(min(data$date_numeric, na.rm = TRUE))


out <- att_gt(
    yname = "bs_score_llm",
    gname = "return_to_office_numeric",
    idname = "firm",
    tname = "date_numeric",
    xformla = ~1,
    data = data,
    est_method = "reg"
)

# Print the summary of the output
summary(out)
