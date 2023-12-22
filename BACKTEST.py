import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from functools import reduce

price_data=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='수정주가', index_col=0)
roe=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='roe', index_col=0)
roa=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='roa', index_col=0)
per=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='per', index_col=0)
pbr=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='pbr', index_col=0)
size=pd.read_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/종목.xlsx', sheet_name='시가총액', index_col=0)

choice=0
n_choice=0
n_0=0
per_0=0
pbr_0=0
roe_0=0
roa_0=0
size_0=0
rebalancing=120
day=0
n_0=0
pbr_0=0
per_0=0
roe_0=0
roa_0=0
size_0=0
per2=0
pbr2=0
roe2=0
roa2=0
size2=0
n_choice=0
per_choice=0
pbr_choice=0
roe_choice=0
roa_choice=0
size_choice=0
while choice!='q':
  choice=input("조건을 선택하세요(종료:q) : n일 이동평균, per, pbr, roe, roa, 시가총액")
  if choice == "n일 이동평균":
    n_choice = input("n을 입력하세요")
    n_0 = input("n일 이상이면 1 이하면 2를 입력하세요")
  elif choice =="per":
    per_choice = input("per을 입력하세요")
    per_0 = input("n 이상이면 1 이하면 2를 입력하세요")
  elif choice == "pbr":
    pbr_choice = input("pbr을 입력하세요")
    pbr_0 = input("n 이상이면 1 이하면 2를 입력하세요")
  elif choice == "roe":
    roe_choice = input("roe를 입력하세요")
    roe_0 = input("n 이상이면 1 이하면 2를 입력하세요")
  elif choice == "roa":
    roa_choice = input("roa를 입력하세요")
    roa_0 = input("n 이상이면 1 이하면 2를 입력하세요")
  elif choice == "시가총액":
    size_choice = input("시가총액을 입력하세요")
    size_0 = input("n 이상이면 1 이하면 2를 입력하세요")
  else:
    pass
rebalancing=input("리밸런싱 주기")
n_choice=int(n_choice)
per_choice=int(per_choice)
pbr_choice=int(pbr_choice)
roe_choice=int(roe_choice)
roa_choice=int(roa_choice)
size_choice=int(size_choice)
rebalancing=int(rebalancing)

# 거래비용
tax = 0.0002 # 현재 거래비용은 0.02%입니다.

# 투자원금
cash = 100000000 # 모의투자 시작금액인 1억으로 시작하겠습니다.
money = cash

risk_free_rate = 0.0359 / 365 # 23년 4월 1일 기준 CD91일물 금리로 하겠습니다.

# 손절기준
loss_cut = 0.99

day = 0
quarter = 0
df_list = '현금 주식총액 포트폴리오가치 일일수익률 총수익률'.split()
backtest = pd.DataFrame(columns = df_list)

print('전체 리밸런싱 횟수는 {}'.format(int(len(price_data)/rebalancing)))
for reb in range(int(len(price_data)/rebalancing)):
    print( reb+1, '회 투자')
    print('현금 : ', money)
    inv = price_data.iloc[day:day+rebalancing,:]
    inv = inv.replace(np.NaN, 0)
    inv = inv.loc[:,inv.iloc[0,:] > 0 ]
    # 스크리닝

    #n일 이동평균
    if n_0 == '1':
        n_0=int(n_0)
        price_mean=price_data.iloc[day-n_0:day].mean()
        price_data2=price_data[day-1:day].mean()>price_mean
        price_data3=pd.DataFrame({'t':price_data2})
        price_data3=price_data3[price_data3>0]
        price_data3=price_data3.dropna(axis=0)
        price_data3=price_data3.T
    elif n_0 == '2':
        n_0=int(n_0)
        price_mean=price_data.iloc[day-n_0:day].mean()
        price_data2=price_data[day-1:day].mean()<price_mean
        price_data3=pd.DataFrame({'t':price_data2})
        price_data3=price_data3[price_data3>0]
        price_data3=price_data3.dropna(axis=0)
        price_data3=price_data3.T
    else:
        price_data3=price_data

    #per
    if per_0 == '1' :    
        per2 = per.iloc[day : day+rebalancing, :]
        per2 = per2.replace(np.NaN, 0)
        per2 = per2.loc[:,per2.iloc[0,:] > per_choice ]
    elif per_0 == '2' :
        per2 = per.iloc[day : day+rebalancing, :]
        per2 = per2.replace(np.NaN, 0)
        per2 = per2.loc[:,per2.iloc[0,:] < per_choice ]
    else:
        per2=per
    
    #pbr
    if pbr_0 == '1' :    
        pbr2 = pbr.iloc[day : day+rebalancing, :]
        pbr2 = pbr2.replace(np.NaN, 0)
        pbr2 = pbr2.loc[:,pbr2.iloc[0,:] > pbr_choice ]
    elif pbr_0 == '2' :
        pbr2 = pbr.iloc[day : day+rebalancing, :]
        pbr2 = pbr2.replace(np.NaN, 0)
        pbr2 = pbr2.loc[:,pbr2.iloc[0,:] < pbr_choice ]
    else:
        pbr2=pbr
    
    #roe
    if roe_0 == '1' :    
        roe2 = roe.iloc[day : day+rebalancing, :]
        roe2 = roe2.replace(np.NaN, 0)
        roe2 = roe2.loc[:,roe2.iloc[0,:] > roe_choice ]
    elif roe_0 == '2' :
        roe2 = roe.iloc[day : day+rebalancing, :]
        roe2 = roe2.replace(np.NaN, 0)
        roe2 = roe2.loc[:,roe2.iloc[0,:] < roe_choice ]
    else:
        roe2=roe
    
    #roa
    if roa_0 == '1' :    
        roa2 = roa.iloc[day : day+rebalancing, :]
        roa2 = roa2.replace(np.NaN, 0)
        roa2 = roa2.loc[:,roa2.iloc[0,:] > roa_choice ]
    elif roa_0 == '2' :
        roa2 = roa.iloc[day : day+rebalancing, :]
        roa2 = roa2.replace(np.NaN, 0)
        roa2 = roa2.loc[:,roa2.iloc[0,:] < roa_choice ]
    else:
        roa2=roa
    
    #시가총액
    if size_0 == '1' :    
        size2 = size.iloc[day : day+rebalancing, :]
        size2 = size2.replace(np.NaN, 0)
        size2 = size2.loc[:,size2.iloc[0,:] > size_choice ]
    elif size_0 == '2' :
        size2 = size.iloc[day : day+rebalancing, :]
        size2 = size2.replace(np.NaN, 0)
        size2 = size2.loc[:,size2.iloc[0,:] < size_choice ]
    else:
        size2=size
    
    #constraint = per2.columns.intersection(pbr2.columns)
    #constraint = constraint.intersection(price_data3.columns)
    #constraint = constraint.intersection(roe2.columns)
    #constraint = constraint.intersection(roa2.columns)
    #constraint = constraint.intersection(size2.columns)
    dataframes = [per2, pbr2, price_data3, roe2, roa2, size2]
    all_columns = set.intersection(*(set(df.columns) for df in dataframes))
    constraint = list(all_columns)
        
    inv_list = price_data[constraint].iloc[0,:].index

    print('투자 후보 갯수는 : ', len(inv_list))

    #스크리닝 끝

    final_inv_list = []
    for i in range(len(inv_list)):
        if inv_list[i] in inv.columns:
            final_inv_list.append(inv_list[i])
        else:
            print(inv_list[i],' 종목이 없습니다')
    print('투자하는 종목의 수는 : ', len(final_inv_list))
    print('투자종목 : ',final_inv_list)


    # 매수 기준 : 동일 비중

    if len(final_inv_list)==0:
        allocation=0
    else:
        allocation = money / len(final_inv_list)


    print('동일 비중 투자 금액은 : ' , allocation)
    final_price_data = inv[final_inv_list].copy()



    vec = pd.DataFrame({'매수수량' : allocation // final_price_data.iloc[0,:]})
    vec = vec.replace(np.NaN, 0)

    #매도(손절 기준)

    loss_cut_money_list = []
    loss_cut_money = 0
    for days in final_price_data.index:
        for stocks in final_price_data.columns:
            if final_price_data.loc[days, stocks] < final_price_data[stocks][0] * ( 1 - loss_cut ):
                loss_cut_money = loss_cut_money + (final_price_data.loc[days, stocks] * float(vec.loc[stocks])) * (1 - tax)
                final_price_data.loc[days:, stocks] = 0

        loss_cut_money_list.append(loss_cut_money)

    product = np.dot(final_price_data, vec)
    product = pd.DataFrame(product)



    balance = pd.DataFrame(index = product.index)
    balance['현금'] = money - product.iloc[0,0]
    balance['현금'] += loss_cut_money_list
    #익절
    #balance['현금'] += profit_cut_money_list


    _backtest = pd.DataFrame(columns = df_list)
    _backtest['주식총액'] = product
    _backtest['현금'] = balance['현금']


    _backtest['포트폴리오가치'] = _backtest['현금'] + _backtest['주식총액']
    _backtest['총수익률'] = (_backtest['포트폴리오가치']/cash)
    _backtest.index = price_data.index[day : day + rebalancing]


    backtest = pd.concat([backtest, _backtest], axis = 0, ignore_index=False)

    money = backtest.iloc[-1,0] + (backtest.iloc[-1,1] * ( 1 - tax))
    day = day + rebalancing
    quarter += 1

backtest['일일수익률'] = backtest['포트폴리오가치'].pct_change()

backtest.to_excel('C:/Users/jerom/OneDrive/바탕 화면/과제/2학년 2학기/기초창의공학설계/백테스트/백테스트 결과_종목.xlsx', index=True)

import yfinance as yf

BM = yf.download('^KS11', start='2013-10-19', end = '2023-10-19')

BM['일일수익률'] = BM['Open'].pct_change()
BM['총수익률'] = BM['Open']/BM['Open'][0]

# CAGR : 연평균 성장률 ( Compound Annual Growth Rate)
num_of_year = int(len(backtest)/365)

CAGR = (backtest['포트폴리오가치'][-1]/cash)**(1/num_of_year) - 1
print('포트폴리오 CAGR은 {} %'.format(CAGR*100))

BENCHMARK_CAGR = ((BM['Open'][len(BM)-1] / BM['Open'][0])) ** (1 / num_of_year) - 1
print("벤치마크 CAGR은 {} %".format(BENCHMARK_CAGR*100))

#MDD : 최저점이 최고점 대비 몇퍼센트의 하락인지?
# (최고점 - 최저점) / 최고점

max_list = [backtest.iloc[0,2]]
min_list = [backtest.iloc[0,2]]

for i in range(len(backtest)):
    if i==0:
        max_list.append(backtest.iloc[0,2])
        min_list.append(backtest.iloc[0,2])

    else:
        if backtest.iloc[i, 2] >= backtest.iloc[i-1,2]:
            max_list.append(backtest.iloc[i, 2])
            min_list.append(backtest.iloc[i, 2])
        else:
            if(max_list[-1]>backtest.iloc[:i,2].max()):
                max_list.append(max_list[-1])
            else:
                max_list.append(backtest.iloc[:i,2].max())
            min_list.append(backtest.iloc[i, 2])


max_list = max_list[1:]
min_list = min_list[1:]
backtest['max'] = max_list
backtest['min'] = min_list
backtest['mdd'] = -((backtest['max'] - backtest['min'])/backtest['max'])

print('BACKTEST MDD는 {}%'.format(backtest['mdd'].min()*100))

backtest['MDD'] = 0
for i in range(len(backtest)):
    if i != 0:
        backtest.iloc[i, -1] = backtest['mdd'][:i].min()
        
#MDD : 최저점이 최고점 대비 몇퍼센트의 하락인지?
# (최고점 - 최저점) / 최고점

max_list = [BM.iloc[0,2]]
min_list = [BM.iloc[0,2]]

for i in range(len(BM)):
    if i==0:
        max_list.append(BM.iloc[0,2])
        min_list.append(BM.iloc[0,2])

    else:
        if BM.iloc[i, 2] >= BM.iloc[i-1,2]:
            max_list.append(BM.iloc[i, 2])
            min_list.append(BM.iloc[i, 2])
        else:
            if(max_list[-1]>BM.iloc[:i,2].max()):
                max_list.append(max_list[-1])
            else:
                max_list.append(BM.iloc[:i,2].max())
            min_list.append(BM.iloc[i, 2])


max_list = max_list[1:]
min_list = min_list[1:]
BM['max'] = max_list
BM['min'] = min_list
BM['mdd'] = -((BM['max'] - BM['min'])/BM['max'])

print('BM MDD는 {}%'.format(BM['mdd'].min()*100))

BM['MDD'] = 0
for i in range(len(BM)):
    if i != 0:
        BM.iloc[i, -1] = BM['mdd'][:i].min()
        
# Sharpe Ratio = (자산 X의 기대수익률 – 무위험 자산 수익률) / 자산 X의 기대수익률의 표준편차
backtest_return = backtest['일일수익률'].mean()
backtest_std = backtest['일일수익률'].std()

BM_return = BM['일일수익률'].mean()
BM_std = BM['일일수익률'].std()

backtest_sharpe = (backtest_return - risk_free_rate) / backtest_std
BM_sharpe = (BM_return - risk_free_rate) / BM_std

print('BACKTEST 샤프 비율은 : ', backtest_sharpe)
print('BENCHMARK 샤프 비율은 : ', BM_sharpe)

fig = make_subplots(rows=4, cols=1,
                    specs=[[{"rowspan":3}],
                          [None],
                          [None],
                          [{}]],
                   shared_xaxes=True,
                   vertical_spacing=0.2,
                   subplot_titles=("수익률","MDD"))

fig.add_trace(go.Scatter(name='코스피 총수익률', x=BM.index, y=BM['총수익률']),
             row=1, col=1)
fig.add_trace(go.Scatter(name='포트폴리오 총수익률', x=backtest.index, y=backtest['총수익률']),
             row=1, col=1)
fig.add_trace(go.Scatter(name='포트폴리오 MDD', x=backtest.index, y=backtest['MDD'], fill='tozeroy'),
             row=4, col=1)
fig.add_trace(go.Scatter(name='BENCHMARK MDD', x=BM.index, y=BM['MDD'], fill='tozeroy'),
             row=4, col=1)

fig.update_layout(height=800, width=1000, plot_bgcolor='rgb(240, 240,240)',
                 title_text="백테스트 결과")

fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

fig.show()
