library(readr)
library(dplyr)

data_main <- read_delim(file = "C:/My Study/A Masters/Data Science/DPV/ASS3/SuperSales/SuperstoreSales_main.csv",
                    delim = ";", col_names = TRUE, col_types = NULL, locale = locale(encoding="ISO-8859-1"))
data_manager <- read_delim(file = "C:/My Study/A Masters/Data Science/DPV/ASS3/SuperSales/SuperstoreSales_manager.csv",
                    delim = ";", col_names = TRUE, col_types = NULL, locale = locale(encoding="ISO-8859-1"))
data_returns <- read_delim(file = "C:/My Study/A Masters/Data Science/DPV/ASS3/SuperSales/SuperstoreSales_returns.csv",
                    delim = ";", col_names = TRUE, col_types = NULL, locale = locale(encoding="ISO-8859-1"))

# make Product table 'product':
product <- data_main %>%
  select("Product Name", "Product Category", "Product Sub-Category") %>%
  rename(name = "Product Name", category = "Product Category", sub_category = "Product Sub-Category") %>%
  arrange(name, category, sub_category) %>%
  group_by(name, category, sub_category) %>%
  distinct() %>%
  ungroup() %>%
  mutate(productid = row_number())

# make Customer table 'customer':
customer <- data_main %>%
  select("Customer Name", Province, Region, "Customer Segment") %>%
  rename(name = "Customer Name", province = Province, region = Region, segment = "Customer Segment") %>%
  arrange(name, province, region, segment) %>%
  group_by(name, province, region, segment) %>%
  distinct() %>%
  ungroup() %>%
  mutate(customerid = row_number())

# make ReturnStatus table
idReturnStatus <- c(0,1) 
returnstatus <- c("Returned", "Not Returned")  
ReturnStatus <- data.frame(idReturnStatus, returnstatus, stringsAsFactors=FALSE)
    
# make Sales table 'sales':
sales <- data_main %>%
  select("Order Date", Sales, "Order Quantity", "Unit Price", Profit, 
         "Shipping Cost", "Customer Name", "Order ID", "Ship Date", "Product Name") %>%
  rename(orderdate = "Order Date", sales = Sales, orderquantity = "Order Quantity", 
         unitprice = "Unit Price", profit = Profit, shippingcost = "Shipping Cost", 
         shipdate = "Ship Date") 

# Combine sales with customerid, productid and returns
sales <- sales %>%
  full_join(customer, by = c("Customer Name" = "name")) %>%
  select( -"Customer Name")
sales <- sales %>%
  full_join(product, by = c("Product Name" = "name")) %>%
  select( -"Product Name")
sales <- sales %>%
  full_join(data_returns, by = c("Order ID" = "Order ID")) %>%
  select( -"Order ID")

# add idReturnStatus to the order returnstatus
sales[c("Status")][is.na(sales[c("Status")])] <- "Not Returned"
sales <- sales %>%
  full_join(ReturnStatus, by = c("Status" = "returnstatus")) %>%
  select( -"Status")


# Determine the data types of a data frame's columns
sapply(sales, class)

# Covert profit, shippingcost variables into numeric variables
sales$profit <- gsub(',', '.', sales$profit)
sales$shippingcost <- gsub(',', '.', sales$shippingcost)
sales$profit <- as.numeric(as.character(sales$profit), stringsAsFactors=FALSE)
sales$shippingcost <- as.numeric(as.character(sales$shippingcost), stringsAsFactors=FALSE)

# Convert character dates into numeric dates
library(lubridate)
sales$orderdate <- dmy(sales$orderdate)
sales$shipdate <- dmy(sales$shipdate)

# check if the product is shipped late or not
sales$diff_in_days<- difftime(sales$shipdate ,sales$orderdate , units = c("days"))
sales$Late <- ifelse(sales$diff_in_days>=3, "late", "not late")

# remove unwanted columns from sales table
sales <- sales[, !(colnames(sales) %in% c("shipdate","province", "region", "segment", 
                                          "diff_in_days", "category", "sub_category"))]

# combining same facts in sales table 
sales <- sales %>%
  arrange(orderdate, customerid, productid, idReturnStatus, Late) %>%
  group_by(orderdate, customerid, productid, idReturnStatus, Late) %>%
  summarise(sales = sum(sales), orderquantity = sum(orderquantity), profit = sum(profit),
            shippingcost = sum(shippingcost)) %>%
  ungroup()


# Filling the Database in PostgreSQL
library(DBI)
library(RPostgreSQL)

drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, port = 5432, host = "castle.ewi.utwente.nl",
                 dbname = "dpv2a009", user = "dpv2a009", password = "qft35ss0",
                 options="-c search_path=ass3")
dbWriteTable(con, "product", value = product, overwrite = T, row.names = F)
dbWriteTable(con, "customer", value = customer, overwrite = T, row.names = F)
dbWriteTable(con, "sales", value = sales, overwrite = T, row.names = F)
dbWriteTable(con, "ReturnStatus", value = ReturnStatus, overwrite = T, row.names = F)

dbGetQuery(con,
           "SELECT table_name FROM information_schema.tables
           WHERE table_schema='ass3'") 
str(dbReadTable(con, c("ass3", "customer")))
str(dbReadTable(con, c("ass3", "product")))
str(dbReadTable(con, c("ass3", "sales")))
str(dbReadTable(con, c("ass3", "ReturnStatus")))