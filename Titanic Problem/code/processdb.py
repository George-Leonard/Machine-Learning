import pandas as pd
import numpy as np

class process:
    def sql(sql):
        pddf = pd.read_sql_table(sql)
        return pddf

    def csv(csv):
        pddf = pd.read_csv(csv)
        return pddf
        
        
    def pandasrenameheaders(pddf, headers):
        headerlist = []
        for n in range(0, len(headers)):
            headerlist.append((headers[n]))
            
        for x in range(0, len(pddf.columns)):
            pddf = pddf.rename(columns={x : headerlist[x]})

        return(pddf)

        
    def nulls(db,null_value):
        nulls = []
        if type(db) == str:
            print("use pandas db")
        else: 
            for i in db:
                for x in range(0, len(db[i])):
                    if db[i].values[x] == null_value:
                        print("NULL AT:"," row: ", x, " column: ", i)
                        nulls.append([x,i])

        return nulls



                    
                    
        


    
