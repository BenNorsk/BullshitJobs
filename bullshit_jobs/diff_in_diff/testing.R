# Install

# Load the libs
library(did)
library(dplyr)

print("Hello world")
# Load the data
data(mpdta)

# Show the data
print(mpdta)
out <- att_gt(
  yname = "lemp",
  gname = "first.treat",
  idname = "countyreal",
  tname = "year",
  xformla = ~1,
  data = mpdta,
  est_method = "reg"
)

summary(out)