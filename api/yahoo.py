import yfinance as yf
import pandas as pd

asset = "MSFT"
start_date = "2024-01-01"
end_date = "2024-12-31"
difference = pd.to_datetime(end_date) - pd.to_datetime(start_date)
print(f'Difference: {difference.days}d')

dat = yf.Ticker(asset)

## Overall
print(f'dat.info: {dat.info}') #Get a dictionary of the company's overall information.
print(f'dat.calendar: {dat.calendar}')

## Financials
print(f'Income Statement: {dat.income_stmt}')
print(f'Quarterly Income Statement: {dat.quarterly_income_stmt}')
print(f'Financials: {dat.financials}') #Get a DataFrame of the company's quarterly financial statements.
print(f'Quarterly Financials: {dat.quarterly_financials}') #Get a DataFrame of the company's quarterly financial statements.
print(f'Balance Sheet: {dat.balance_sheet}') #Get a DataFrame of the company's balance sheet.
print(f'Quarterly Balance Sheet: {dat.quarterly_balance_sheet}') #Get a DataFrame of the company's quarterly balance sheet.
print(f'Cash Flow: {dat.cashflow}') #Get a DataFrame of the company's cash flow.
print(f'Quarterly Cash Flow: {dat.quarterly_cashflow}') #Get a DataFrame of the company's quarterly cash flow.
print(f'Earnings: {dat.earnings}')
print(f'Quarterly Earnings: {dat.quarterly_earnings}') #Get a DataFrame of the company's quarterly earnings.

## Other Financials
print(f'dat.analyst_price_targets: {dat.analyst_price_targets}')
print(f'Institutional Holders: {dat.institutional_holders}') #Get a DataFrame of the company's institutional shareholders.
print(f'Major Holders: {dat.major_holders}') #Get a DataFrame of the company's major shareholders.
print(f'Insider Transactions: {dat.insider_transactions}') #Get a DataFrame of the company's insider transactions.

## Other
print(f'Sustainability: {dat.sustainability}') #Get a DataFrame of the company's sustainability.
print(f'Recommendations: {dat.recommendations}') #Get a DataFrame of the company's recommendations.

#print(f"dat.history(period=\'1mo\'): {dat.history(period='1mo')}")
#print(f'dat.option_chain(dat.options[0]).calls: {dat.option_chain(dat.options[0]).calls}')

