import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('PVF020_Baseline.csv')

print(data.head())

data_sorted = data[['Country name', '2011']].sort('2011', ascending = True).dropna()
print(data_sorted.head(5))

temp = data_sorted.head(5)

plt.style.use('grayscale')
plt.yticks([500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
plt.bar(range(len(temp['2011'])),temp['2011'].values)
plt.xticks(range(len(temp['2011'])), temp['Country name'].values,size='large', rotation=70)
plt.ylabel('kilo calory per capita per day',weight='bold', size='large')
plt.title('Lowest per Capita Food Supply\n',weight='heavy',size = 'x-large')
plt.ylim([0,4000])
#plt.show()

plt.figure(figsize=(12,5))
#plt.figure(122)

#Plot 1
plt.subplot(121)
temp2 = data_sorted.head(5)
plt.style.use('grayscale')
plt.yticks([500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
plt.bar(range(len(temp2['2011'])),temp2['2011'].values)
plt.xticks(range(len(temp2['2011'])), temp2['Country name'].values,size='large', rotation=70)
plt.ylabel('kilo calory per capita per day',weight='bold', size='large')
plt.title('Lowest per Capita Food Supply\n',weight='heavy',size = 'x-large')
plt.ylim([0,4000])



#Plot 2
plt.subplot(122)
temp = data_sorted.tail(5)
plt.style.use('grayscale')
plt.yticks([500,1000,1500,2000,2500,3000,3500,4000,4500,5000])
plt.bar(range(len(temp['2011'])),temp['2011'].values)
plt.xticks(range(len(temp['2011'])), temp['Country name'].values,size='large', rotation=70)
plt.ylabel('kilo calory per capita per day',weight='bold', size='large')
plt.title('Highest per Capita Food Supply\n',weight='heavy',size = 'x-large')
plt.ylim([0,4000])

plt.show()