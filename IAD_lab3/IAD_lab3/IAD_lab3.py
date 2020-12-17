
import pandas as pnd
import seaborn as sn
import matplotlib.pyplot as ploting
from sklearn.linear_model import LinearRegression
from itertools import islice
from sklearn.metrics import r2_score

pnd.options.display.max_rows = 1000
pnd.options.display.max_columns = 1000


def lag_corr(series1: pnd.Series, series2: pnd.Series, lag: int):
    if lag == 0 :
        return series1.corr(series2)
    shifted = series1.shift(lag)
    shifted.dropna()
    return shifted.iloc[abs(lag):-abs(lag)].corr(series2.iloc[abs(lag):-abs(lag)])


#%%

def best_lag(series1: pnd.Series, series2: pnd.Series, max_lag: int):
    best_corr = lag_corr(series1, series2, 0)
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        corr = lag_corr(series1, series2, lag)
        if corr > best_corr:
            best_сorr, best_lag = corr, lag
    return best_corr, best_lag


url = 'https://raw.githubusercontent.com/VasiaPiven/covid19_ua/master/covid19_by_area_type_hosp_dynamics.csv'
value = pnd.read_csv(url, error_bad_lines=False)
value1 = value

value1.drop(columns=['new_susp', 'new_confirm', 'new_death' ,'new_recover'], inplace=True)


value1 = value1.groupby(['zvit_date', 'registration_area']).sum()


value1.reset_index(inplace=True)


value1.set_index('zvit_date', inplace=True)


# data.head()


new_df = pnd.DataFrame()


for area in value1.registration_area.unique():
    new_df[area] = value1.loc[value1.registration_area == area].active_confirm


new_df = new_df.fillna(0)

new_df.corr()

corr_best = new_df.corr(method=lambda x, y: best_lag(pnd.Series(x), pnd.Series(y), max_lag=30)[0])
# print(corr_best)

corr_lag = new_df.corr(method=lambda x, y: best_lag(pnd.Series(x), pnd.Series(y), max_lag=30)[1])
print(corr_lag.head())


maxim = "Сумська"




def lagANDcorr(row, column, lag):
    maxlag, maxcor = 0, row.corr(column)
    for cur_lag in range(1, lag + 1):
        cur_corr = row.corr(column.shift(cur_lag))
        if cur_corr > maxcor:
            maxlag = cur_lag
            maxcor = cur_corr
    return -1*maxlag, maxcor





active_confirm_by_state = pnd.DataFrame()

all_states = value['registration_area'].unique()

for state in all_states:
    cur_state = [active_confirm_by_state, value.loc[value['registration_area'] == state].groupby(
        'zvit_date').sum()['active_confirm'].to_frame(state)]

    active_confirm_by_state = pnd.concat(cur_state, axis=1)

active_confirm_by_state = active_confirm_by_state.sort_index().fillna(0.0)
active_confirm_by_area_correlation = active_confirm_by_state.corr()




lag_df = pnd.DataFrame(0, columns=all_states, index=all_states)
lag_corr_df = pnd.DataFrame(0.0, columns=all_states, index=all_states)
for state1 in all_states:
    for state2 in all_states:
        lag, corr = lagANDcorr(active_confirm_by_state[state1], active_confirm_by_state[state2], 100)
        lag_df.at[state1, state2] = lag
        lag_corr_df.at[state1, state2] = corr

# print(lag_corr_df)




sn.heatmap(lag_df, cmap="YlGnBu")
ploting.show()
sn.heatmap(lag_corr_df,vmin=0.85, vmax=1, cmap="YlGnBu")
ploting.show()

# leader = active_confirm_by_area.iloc[-1:].idxmax(axis=1)[0]
# test_lag = lag_table.at[leader, 'Львівська']
# shifter = active_confirm_by_area['Львівська'].shift(test_lag).fillna(0)

col = lag_df.columns.tolist()
count = 0
lider_dynamic = ''



for i in col:
    temp1 = 0
    temp2 = 0
    for j in col:
        if lag_df[i][j] > 0:
            temp1 += 1
        elif lag_df[i][j] < 0:
            temp2 += 1

        if temp1>count:
            count = temp1
            lider_dynamic = i
        elif temp2>count:
            count = temp2
            lider_dynamic = i
print()
print()
print("Лідер по динаміці: ", lider_dynamic)
print()
print()


print()
print()
maxim1 = lag_corr_df[lider_dynamic].idxmax()
print(maxim)
print()
print()

def prediction(data, values_for_x, values_for_y, days):
    # заповненя дааними
    reg = pnd.DataFrame()
    reg_lag = lag_df[values_for_x][values_for_y]
    reg['x'] = data[values_for_x].shift(reg_lag)
    reg['y'] = data[values_for_y]
    reg = reg.dropna()

    # представлення даних
    x1 = list(reg['x'])
    x = [[] for i in range(len(x1))]
    for i in range(len(x1)):
        x[i].append(x1[i])

    # вибір тестових і тренінгових даних
    x_train = x[:-days]
    x_test = x[-days:]
    y_train = reg['y'][:-days]
    y_test = reg['y'][-days:]

    # побудова передбачень
    regr = LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)


    print('coeff of determination:')
    print(r2_score(y_test,y_pred))


    # plt.scatter(x_test, y_test, color ='pink')
    # plt.plot(x_test, y_pred, color = 'r')
    # plt.xlabel(values_for_x)
    # plt.ylabel(values_for_y)
    # plt.show()
    return y_pred

pred = pnd.DataFrame()
days =100
for i in col:
    pred[lider_dynamic] = active_confirm_by_state[lider_dynamic][-days:]
    pred[i] = prediction(active_confirm_by_state, lider_dynamic, i, days)
# print(pred)

# exit()

def convert(lst, var_lst):
    it = iter(lst)
    return [list(islice(it, i)) for i in var_lst]


var_lst = [5 for i in range(5)]
col_3_9 = convert(col, var_lst)
# col_3_9


for i in range(5):
    for j in range(5):
        ploting.plot(pred.index, pred[(col_3_9[i][j])], color = 'black')
        ploting.plot(pred.index, active_confirm_by_state[(pred.index[0]):(pred.index[-1])][(col_3_9[i][j])], color = 'blue')
        ploting.xticks(rotation=90)
        ploting.title(col_3_9[j][i])
        ploting.show()
        # ploting.savefig(f'foo{i}.png')
