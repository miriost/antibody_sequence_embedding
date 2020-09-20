# extract V family
library(alakazam)
library(dplyr)

args = commandArgs(trailingOnly=TRUE)
if (length(args)<2 || length(args)>3) {
  stop("Usage: V_family_lables.R <input csv file> <output file>", call.=FALSE)
}
input_file = args[1]
output_file = args[2]

print(paste('loading input from', input_file))
db <- read.csv(input_file, sep='\t')
db['V_FAMILY'] <- getFamily(db$V_CALL, first = FALSE, collapse = TRUE)
write.table(db, file = input_file, append = FALSE, quote = FALSE, sep = '\t',
                          eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                          col.names = TRUE, qmethod = c("escape", "double"),
                          fileEncoding = "")
write.table(db %>% select(V_FAMILY), file = output_file, append = FALSE, quote = FALSE, sep = '\t',
                          eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                          col.names = TRUE, qmethod = c("escape", "double"),
                          fileEncoding = "")
print(paste('labels file saved to', output_file))

