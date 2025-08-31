rm(list=ls())
gc()

library(dplyr)
library(mongolite)
library(dotenv)

load_dot_env('.env')

mongo_uri = Sys.getenv('MONGO_URI')

data <- read.csv('C:\\Users\\moham\\Desktop\\New folder\\Procurement\\R\\Procurement KPI Analysis Dataset.csv')
nrow(data)

mongo <- mongo(
    collection= 'Procurement-AI-Agent',
    db= 'test',
    url = mongo_uri
)

mongo$insert(data)
