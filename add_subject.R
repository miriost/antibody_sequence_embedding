
args = commandArgs(trailingOnly=TRUE)
if (length(args)<1 || length(args)>1) {
  stop("Usage: add_subject.R <tab file>", call.=FALSE)
}
file_path = args[1]

print(paste("file:", file_path, sep=" "))

df = read.csv(file_path, sep='\t', stringsAsFactor=F)
df['SUBJECT'] = sapply(df[,'FILENAME'], function(x) strsplit(x, '_S')[[1]][1])
write.table(df, row.names = FALSE, quote = FALSE, file_path, sep='\t')
