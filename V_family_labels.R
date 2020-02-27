# extract V family
library(alakazam)
library(dplyr)

# Subset example data
path <-'/media/miri-o/Documents/filtered_data_sets/'
db <- read.csv(paste(path,'FLU_data_012119_HCV_model_Jan19_2019_FILTERED_DATA.csv', sep = ""))

a <- getFamily(db$V_CALL, first = FALSE, collapse = TRUE)
db$V_FAMILY <- a
write.table(a, file = paste(path, "FLU_data_HCV_model_Jan21_2019_labels.csv", sep = ""), append = FALSE, quote = FALSE, sep = "\t",
                          eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                          col.names = TRUE, qmethod = c("escape", "double"),
                          fileEncoding = "")
