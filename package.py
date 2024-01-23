# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 21:01:32 2022

@author: shwet
"""
import pandas as pd # tool for data processing
from pandas_datareader import data # tool for data processing
import matplotlib.pyplot as plt
import plotly.graph_objects as go # tool for data visualization
import yfinance as yf # tool for downloading histrocial market data from "Yahoo! Finance"
from datetime import date # tool for manipulating dates and times
from dateutil.relativedelta import relativedelta # tool for manipulating dates and times
import numpy as np # tool for handling vectors, matrices or large multidimensional arrays
from sklearn.linear_model import LinearRegression # tool for machine leraning (Linear Model)
from sklearn.model_selection import train_test_split # tool for machine learning
import plotly.io as pio
import matplotlib.dates as mdates
import quandl
pio.renderers.default = "browser"

sdate = '2012-05-06'
edate = '2022-05-07'

while True:
    try:
        
        symbol = input('Please enter a stock symbol: ')
        df = data.DataReader(symbol, 'yahoo', sdate, edate)
        break
    
    except(KeyError):
        print(">>> {0} is not a valid stock symbol. Please try again...".format(symbol) )




print()
print('OPTIONS FOR ANALYSING STOCK DATA:')
choice = True
while choice:
    print("What would you like to do: \n1 - Show the price chart of chosen stock\n2 - Show price comparison with an additional stock\n3 - Show the revenue and earnings of chosen stock\n4 - Show the cash flow statement of chosen stock\n5 - Show the analyst recommendations for the stock of the last 6 months\n6 - Show the price prediction\n0 - Quit the program\n")
    choice = int(input("Please enter your choice :"))
    if choice ==1:
        sdate = input('Please define startdate (in YYYY-MM-DD format): ')
        edate = input('Please define enddate (in YYYY-MM-DD format): ')
        df = data.DataReader(symbol, 'yahoo', sdate, edate)
        df = df.reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date,
                                 y=df['Adj Close']))
        
        fig.update_layout(title=f'{symbol} Stock Price from {sdate} to {edate}',
                          yaxis_title='Adjusted Closing Price in USD',
                          )

        fig.show()
  
    # OPTION 2: Show me a price comparison with an additional stock
    elif choice == 2:
        print()
        sdate = input('Please define startdate (in YYYY-MM-DD format): ')
        edate = input('Please define enddate (in YYYY-MM-DD format): ')
    
        df = data.DataReader(symbol, 'yahoo', sdate, edate)
        df = df.reset_index()
        
        while True:
            try:
                symbol_2 = input("\nWith which stock would you like to compare the {0} stock? \nPlease enter the respective stock symbol: ".format(symbol))
                df_2 = data.DataReader(symbol_2, 'yahoo', sdate, edate)
                df_2 = df_2.reset_index()
                break
            except(KeyError):
                print("{0} is not a valid stock symbol. Please try again...".format(symbol_2))

                
        # Plotting the chart
        fig = go.Figure()
        # Add the data from the first stock
        fig.add_trace(go.Scatter(
                    x=df.Date,
                    y=df['Adj Close'],
                    name="{0} Stock".format(symbol)))
        
        # Add the data from the second stock
        fig.add_trace(go.Scatter(
                    x=df_2.Date,
                    y=df_2['Adj Close'],
                    name="{0} Stock".format(symbol_2)))
        fig.update_layout(title="Price Comparison of {0} Stock and {1} Stock from {2} to {3}".format(symbol,symbol_2,sdate,edate), 
                          yaxis_title='Adjusted Closing Price in USD')
        
        fig.show()
    
     #  OPTION 3: Show me the revenue and earnings of my chosen stock
    elif choice == 3:
        try:
            stockvariable = yf.Ticker(symbol)
            df = stockvariable.earnings
            df = df.reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df.Year,
                        y=df.Revenue,
                        name='Revenue',
                        marker_color='purple'))
            fig.add_trace(go.Bar(x=df.Year,
                        y=df.Earnings,
                        name='Earnings',
                        marker_color='orange'))
            fig.update_layout(
            title="Revenue and Earnings of {0} Stock".format(symbol),
            xaxis_tickfont_size=14,
            yaxis=dict(
                title='USD (billions)',
                titlefont_size=16,
                tickfont_size=14),
            barmode='group',
            bargap=0.15,   
            bargroupgap=0.1)

            fig.show()
   
        except(ValueError,AttributeError):
            print('\n>>> Unfortunately, there is no data provided for your stock!') 
        
        
        #  OPTION 4: Show me the cash flow statement of my chosen stock
    elif choice == 4:
        try:
            stockvariable = yf.Ticker(symbol)
            cashflow = stockvariable.cashflow
            cashflow = cashflow.rename(columns=lambda x: str(x.year))
            ICF = cashflow.loc['Total Cashflows From Investing Activities',]
            FCF = cashflow.loc['Total Cash From Financing Activities',]
            OCF = cashflow.loc['Total Cash From Operating Activities',]
            NI = cashflow.loc['Net Income',]
            CF = [OCF, ICF, FCF, NI] 
            cashflow = pd.DataFrame(CF)
            cashflow = cashflow.reset_index()
            cashflow = cashflow.rename(columns={'index': 'Cashflows'})
            cashflow.replace('Total Cash From Operating Activities', 'Cash Flow From Operating Activities', inplace=True)
            cashflow.replace('Total Cashflows From Investing Activities', 'Cash Flow From Investing Activities', inplace=True)
            cashflow.replace('Total Cash From Financing Activities', 'Cash Flow From Financing Activities', inplace=True)
            first_cf = cashflow.iloc[:,1]
            second_cf = cashflow.iloc[:,2]
            third_cf = cashflow.iloc[:,3]
            fourth_cf = cashflow.iloc[:,4]
          # Plotting the bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                    x = cashflow.Cashflows,
                    y = first_cf,
                    name = first_cf.name,
                    marker_color="yellow"))
            
            fig.add_trace(go.Bar(
                    x = cashflow.Cashflows,
                    y = second_cf,
                    name = second_cf.name,
                    marker_color="orange"))
            
            fig.add_trace(go.Bar(
                    x = cashflow.Cashflows,
                    y = third_cf,
                    name = third_cf.name,
                    marker_color="red"))
    
            fig.add_trace(go.Bar(
                    x = cashflow.Cashflows,
                    y = fourth_cf,
                    name = fourth_cf.name,
                    marker_color="pink"))
    

            fig.update_layout(barmode = 'group',
                              bargap = 0.15,
                              bargroupgap = 0.1,
                              title = "Cash Flow Statement of {0} Stock".format(symbol),
                              xaxis_tickfont_size = 14,
                              xaxis_tickangle =0,
                              yaxis = dict(
                                        title = 'USD (billions)',
                                        titlefont_size = 14,
                                        tickfont_size = 12))
                              
            fig.show()
            
            
        except(ValueError,AttributeError):
            print('\n>>> Unfortunately, there is no data provided for your stock!') 

    elif choice == 5:
        try:
            today = date.today()
            today = today.strftime('%Y-%m-%d')
            six_months = date.today() - relativedelta(months=+6)
            six_months = six_months.strftime('%Y-%m-%d')
        
            df2 = yf.Ticker(symbol)
            rec = df2.recommendations
            rec = rec.loc[six_months:today,]
            if rec.empty:
                print("\n>>>Unfortunately, there are no recommendations by analysts provided for your chosen stock!")
                    
            else:    
                rec = rec.reset_index()
                rec.drop(['Firm', 'From Grade', 'Action'], axis=1, inplace=True)
                rec.columns = (['date', 'grade'])
                rec['value'] = 1
                rec = rec.reset_index()
                fig = go.Figure(data=[go.Pie(labels=rec.grade,values=rec.value)])
                fig.update_layout(title_text="Analyst Recommendations of {0} Stock from {1} to {2}".format(symbol,six_months,today))

                # Plotting the chart
                fig.show()  

        except(ValueError,AttributeError):
            print('\n>>>Unfortunately, there are no recommendations provided for your chosen stock!') 

# OPTION 6
    elif choice == 6:
       data = quandl.get('WIKI/{0}'.format(symbol), start_date=sdate, end_date=edate)# Load data from Quandl
       data.to_csv('stock_market.csv')# Save data to CSV file
       df = pd.DataFrame(data, columns=['Close'])# Create a new DataFrame with only closing price and date
       df = df.reset_index()# Reset index column so that we have integers to represent time 

       years = mdates.YearLocator() # Get every year
       yearsFmt = mdates.DateFormatter('%Y') # Set year format
       fig,ax= plt.subplots()# Create subplots to plot graph and control axes
       ax.plot(df['Date'], df['Close'])
       # Format the ticks
       ax.xaxis.set_major_locator(years)
       ax.xaxis.set_major_formatter(yearsFmt)

       plt.title('Close Stock Price History [{0} - {1}]'.format(sdate,edate))
       plt.xlabel('Date')
       plt.ylabel('Closing Stock Price in $')
       plt.show()

       train, test = train_test_split(df, test_size=0.20,random_state=0)

       # Reshape index column to 2D array for .fit() method
       X_train = np.array(train.index).reshape(-1, 1)
       y_train = train['Close']

       model = LinearRegression()
       model.fit(X_train, y_train)

       #Train set graph
       plt.title('Linear Regression | Price vs Time')
       plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
       plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
       plt.xlabel('Date')
       plt.ylabel('Stock Price')
       plt.legend()
       plt.show()

       # Create test arrays
       X_test = np.array(test.index).reshape(-1, 1)
       y_test = test['Close']

       # Generate array with predicted values
       y_pred = model.predict(X_test)
       year=int(input("Enter the year to predict stock price:"))
       print("\n\nPredicted stock market price of year {0} is:{1}\n\n".format(year,model.predict([[year]])))

 

 # OPTION 7: Quit the program
    elif choice == 0:
        print('\n')
        choice = None
        
    # If user inputs a non valid option
    else:
        print('\n> Your input is not a given choice. Please try again...')