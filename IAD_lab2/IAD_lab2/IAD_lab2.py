# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gmplot
import openpyxl

# %%
regions = ['Вінницька', 'Волинська', 'Дніпропетровська', 'Донецька', 'Житомирська',
    'Закарпатська', 'Запорізька', 'Івано-Франківська', 'Київська',
    'Кіровоградська', 'Луганська', 'Львівська', 'Миколаївська', 'м. Київ',
    'Одеська', 'Полтавська', 'Рівненська', 'Сумська', 'Тернопільська',
    'Харківська', 'Херсонська', 'Хмельницька', 'Черкаська', 'Чернівецька',
    'Чернігівська']
def print_regions(regions):
    for i in range(len(regions)):
        print(f"{i+1}. {regions[i]} область")
print_regions(regions)
choosed_region = int(input('Оберіть регіон зі списку для якого завантажити данні - '))
choosed_region-=1

# %%

url = 'https://raw.githubusercontent.com/VasiaPiven/covid19_ua/master/covid19_by_area_type_hosp_dynamics.csv'

data_frame = pd.read_csv(url)
data_frame_for_choosed_region = data_frame.loc[data_frame['registration_area'] == regions[choosed_region]]
sum_by_date = data_frame_for_choosed_region.groupby('zvit_date').sum()


# %%

plt.plot(sum_by_date['new_susp'], label='підозрілі')
plt.plot(sum_by_date['new_confirm'], label='підтвердженні')
plt.plot(sum_by_date['active_confirm'], label='активні')
plt.plot(sum_by_date['new_death'], label='померли')
plt.plot(sum_by_date['new_recover'], label='одужали')
plt.title(regions[choosed_region]+' область')
plt.legend()
plt.xlabel('Дата')
plt.ylabel('Кількість')
length = sum_by_date.count()+1
plt.xticks(np.arange(0, length[1], 10))
plt.xticks(rotation=270)
plt.show()


# %%
print_regions(regions)
print('Оберіть регіони для які порівняти: ')
reg1, reg2 = map(int, input().split())
params = ['new_susp', 'new_confirm', 'active_confirm','new_death','new_recover']
for i in range(len(params)):
    print(f'{i}. {params[i]}')
n = int(input('Оберіть параметр за яким порівнювати: '))
region1 = data_frame.loc[data_frame['registration_area'] == regions[reg1]]
sum_by_date_1 = region1.groupby('zvit_date').sum()
region2 = data_frame.loc[data_frame['registration_area'] == regions[reg2]]
sum_by_date_2 = region2.groupby('zvit_date').sum()


# %%
plt.plot(sum_by_date_1[params[n]], label=regions[reg1])
plt.plot(sum_by_date_2[params[n]], label=regions[reg2])
plt.legend()
plt.title(params[n])
plt.xlabel('Дата')
plt.ylabel('Кількість')
length = sum_by_date.count()+1
plt.xticks(np.arange(0, length[1], 10))
plt.xticks(rotation=270)
plt.show()


# %%
data_frame_for_geo = pd.read_csv('https://raw.githubusercontent.com/VasiaPiven/covid19_ua/master/covid19_by_settlement_actual.csv')
gmap2 = gmplot.GoogleMapPlotter(48, 31, zoom=7)
gmap2.scatter(data_frame_for_geo['registration_settlement_lat'], data_frame_for_geo['registration_settlement_lng'], 'red',
                              size = 1000, marker = False)
gmap2.draw('map.html')

with pd.ExcelWriter('output.xlsx') as writer:
        data_frame_for_geo.to_excel(writer, sheet_name='Map')
        sum_by_date_1.to_excel(writer, sheet_name=regions[reg1])
        sum_by_date_2.to_excel(writer, sheet_name=regions[reg2])
        sum_by_date.to_excel(writer, sheet_name=regions[choosed_region])