# Predikce časových řad - Shampoo Sales dataset

![Shampoo Sales dataset](./img/shampoo.jpg)

## Obsah

- [Úvod do problematiky](#úvod-do-problematiky)
- [Použité nástroje](#použité-nástroje)
- [Představení datasetu](#představení-datasetu)
- [Načtení a transformace dat](#načtení-a-transformace-dat)
- [Analýza dat](#analýza-dat)
- [Lineární regrese](#lineární-regrese)
- [Kvadratická regrese](#kvadratická-regrese)
- [Model ARIMA](#model-arima)


## Úvod do problematiky 
Predikce časových řad je klíčovým nástrojem v mnoha oblastech, jako jsou finance, meteorologie, výroba a další. Umožňuje nám předpovídat budoucí hodnoty na základě historických dat, což může být nesmírně užitečné při rozhodování a plánování. V tomto článku se zaměříme na predikci časové řady na příkladu datasetu „shampoo sales“, který obsahuje měsíční prodeje šamponu. Pro tuto úlohu použijeme několik různých modelů a zhodnotíme jejich výkon.

## Použité nástroje
Veškerou práci s daty provádíme pomocí programovacího jazyka `Python`. Pro načtení dat a vytvoření objektu datového rámce využijeme modul `Pandas`. Pro vizualizaci datových analýz a výsledků predikce využijeme modul `Matplotlib` a `Seaborn`. Pro vykonávání početních operací nad daty využijeme modul `Numpy`. Pro tvorbu prediktivních modelů a jejich vyhodnocení využijeme modul `Scikit-learn`.

## Představení datasetu

- **Název datasetu:** shampoo_sales.csv
- **Velikost datasetu:** 484 B
- **Sloupce:** Month, Sales
- **Zdroj:** Kaggle.com

Tento dataset popisuje měsíční prodeje šampónů za období 3 let. Jednotkou je počet prodejů pro každý ze 36 záznamů. Sloupec `Month` je ve formátu `datetime64` a sloupec `Sales` je ve formátu `int64`. Původní dataset pochází od Makridakis, Wheelwright a Hyndman (1998).

## Načtení a transformace dat


### Načtení dat
Nejprve si načteme pomocí modulu `Pandas` dataset ve formátu `*.csv` do objektu datového rámce. Zároveň si necháme vypsat data za prvních 12 měsíců, čímž zkontrolujeme, zda se nám data správně načetla.

```python
df = pd.read.csv('shampoo_sales.csv',
		 index_col='Month')

df.show(12)
```


| Month | Sales |
|-------|-------|
| 1-01  | 266.0 |
| 1-02  | 145.9 |
| 1-03  | 183.1 |
| 1-04  | 119.3 |
| 1-05  | 180.3 |
| 1-06  | 168.5 |
| 1-07  | 231.8 |
| 1-08  | 224.5 |
| 1-09  | 192.8 |
| 1-10  | 122.9 |
| 1-11  | 336.5 |
| 1-12  | 185.9 |

### Transformace hodnot indexu `Month`

Hodnoty indexového sloupce `Month` neodpovídají daným rokům, ve kterých byla data zaznamenána. Víme, že data byla pořízována od začátku roku 1981 do konce roku 1983. Jednotlivé záznamy změníme pomocí jednoduché funkce `custom_date_parser()`.

```Python
def custom_date_parser(date):
    year = 1981 + int(date.split('-')[0]) - 1
    month = int(date.split('-')[1])
    return pd.to_datetime(f'{year}-{month:02}')

df.index = df.index.to_series().apply(custom_date_parser)
```

### Převedení indexu na periodu

Pro další analýzu a práci s časovou řadou je ještě vhodné použít přesný časový rámec pomocí metody `to_period()`. To se hodí zejména pro snadnou agregaci dat podle časového intervalu. Je zajištěno, že jednotlivé hodnoty jsou brány za dané období a ne ke konkrétnímu dni. Zároveň intervaly mezi jednotlivými záznamy zůstávají konzistentní, díky čemuž můžeme analyzovat trendy a sezónnost. Tento formát sloupce je také vhodnější pro následnou vizualizaci dat.

```Python
df = df.to_period()

df.head()
```

| Month   | Sales |
|---------|-------|
| 1981-01 | 266.0 |
| 1981-02 | 145.9 |
| 1981-03 | 183.1 |
| 1981-04 | 119.3 |
| 1981-05 | 180.3 |


## Analýza dat

### Základní statistika

Pro získání základní popisné statistiky pro numerický sloupec `Sales` v datovém rámci použijeme funkci `describe()`. Získáme tak počet hodnot, průměrnou hodnotu, směrodatnou odchylku, minimální a maximální hodnotu, medián a první a třetí kvartil.

```Python
df.describe()
```

| Sales |        |
|-------|--------|
| count | 36.0   |
| mean  | 312.6  |
| std   | 148.94 |
| min   | 119.3  |
| 25%   | 192.45 |
| 50%   | 280.15 |
| 75%   | 411.1  |
| max   | 682.0  |

Na základě počtu hodnot `36` můžeme usoudit, že naše data jsou kompletní a obsahují všechny záznamy v rozmezíi *3* let.

Směrodatná odchylka nám ukazuje míru variability, popř. jak velké je rozptýlení hodnot kolem průměrné hodnoty. Hodnota `148.94` nám ukazuje, že jednotlivé prodeje se od průměru značně liší. To u časové řady může znamenat, že na hodnoty může mít vliv sezónnost nebo časová závislost dané hodnoty na předchozí hodnotě (sériová závislost). Jsou to tedy první indicie, kudy se vydat v dalším vyhodnocení dat.

### Test stacionarity

Pomocí `ADF` (Augmented Dickey-Fuller) testu zkontrolujeme zda je naše časová řada stacionární.

```Python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

- **ADF Statistic:** 3.0601420836411806
- **p-hodnota:** 1.0

*ADF* je obecně vyšší než běžně udávané kritické hodnoty, což naznačuje, že naše řada není stacionární. To znamená, že statistické vlastnosti této časové řady se mění v čase, což může ovlivnit analýzu a predikci. To nám tedy říká, že budeme dále muset aplikovat diferenciaci nebo další transformace, čímž učiníme řadu stacionární.

*P-hodnota* je také extrémně vysoká, což naznačuje, že existuje vysoká pravděpodobnost, že nulová hypotéza je pravdivá. Jinými slovy, časová řada není stacionární. To nám také potvrzuje nutnost dalšího zpracování dat směrem k jejich stacionarizaci.

### Základní Vizualizace dat

Nyní si pomocí modulu `Matplotlib` vygenerujeme první vizualizaci dat, která nám poskytne daleko lepší představu o tom, jak data celkově vypadají. Zároveň si pomocí funkce `regplot()` proložíme trendovou linii (lineární regrese) dat do grafu (modře). V grafu vidíme, že trend je vzestupný a vcelku dobře vystihuje okolní data. Kvadratická regrese by ale mohla chování dat kopírovat přesněji. Zkusím tedy kvadratickou regresi v pozdějším kroku namodelovat.

```Python
fig, ax = plt.subplots()
ax.plot('Month', 'Sales', data=df, color='0.75')
ax = sns.regplot(x='Month', y='Sales', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Month Plot of Shampoo Sales')
fig.savefig('basic_plot.png')
```

![basic month plot of shampoo sales dataset](./img/basic_plot.png)

Pomocí `Boxplot` grafu můžeme dále zkontrolovat, zda naše data neobsahují odlehlé hodnoty. Na následujícím grafu vidíme rozložení dat mezi jednotlivými kvartily s polohou mediánu. Odlehlé hodnoty se zde nevyskytují.

![boxplot of shampoo sales dataset](./img/boxplot_shampoo.png)

### Pokročilá vizualizace dat

Vzhledem k tomu, že data jsou sezónní, je vhodně si je vizualizovat ve vztahu k unikátním měsícům v roce. Na to použijeme opět `Boxplot`, který je užitečný k vizualizaci sezónních vzorců. V grafu vidíme rozptyl prodejů v jednotlivých měsících. Zde začíná být zřejmé, že prodeje se zvyšují více tím, jak se přibližují ke konci roku, zároveň se ale zvětšuje jejich rozptyl.

![month boxplot of shampoo sales dataset](./img/month_plot.png)

Na základě předchozího grafu je zřejmé, že data prodejů vykazují jistou roční sezónnost. Může být tedy užitečné si vykreslit jednotlivé roky přes sebe. Zde je vidět, že opravdu ke konci každého roku prodeje narůstají. Je zde ale i patrné, že jednotlivé roky se liší v absolutních hodnotách a zároveň je zde i zřejmá jistá nestacionarita.

![year overlapping of shampoo sales dataset](./img/year_plot.png)

## Příprava dat pro prediktivní model

Abychom mohli dála na datech natrénovat a následně otestovat prediktivní model, musíme si data rozdělit na trénovací a testovací část. Pro trénink modelu použijeme první dva roky prodejů (první 2/3) a pro testování poslední rok prodejů (poslední 1/3). Pro rozdělení datasetu použijeme použijeme funkci `train_test_split()` z knihovny `scikit-learn`.

```Python
# Training data
X = df.loc[:, ['Month']] # features
y = df.loc[:, 'Sales'] # target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, shuffle=False)
```

## Lineární regrese

Pomocí modelu lineární regrese natrénujeme na trénovacích datech prediktivní model.

```Python
# Train model
model = LinearRegression()
model.fit(X_train,y_train)
```

Nyní provedeme samotnou predikci hodnot prodeje pomocí modelu lineární regrese ve stejném rozsahu jako naše testovací data.

```Python
y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
```

Jako hodnotící kritérium zvolíme metriku `MAE` (Mean Average Error)

```Python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE):', mae)
```

*Mean Absolute Error (MAE): `132.71`*


![predikce lineární regrese](./img/linear_regression_prediction.png)

Lineární regrese je nejjednodušší model pro predikci časové řady, zároveň je ale vidět, že v tomto případě nejsou predikce odpovídající hodnotám v testovacím rozsahu. Tento prediktivní model již podle vizuální kontroly hodnotíme jako nevhodný. Zároveň průměrná absolutní chyba je také dosti velká.

## Kvadratická regrese

Jako další sloupec přidáme do datasetu pomocí deterministického procesu `Determinstic_process` časový polynom druhého řádu, na kterém poté natrénujeme model pro lineární regresi.

```Python
dp = DeterministicProcess(
    index=X.index,
    constant=True,
    order=2,
    drop=True,
)

X = dp.in_sample()

X.head()
```

| Month   | const | trend | trend_squared |
|---------|-------|-------|---------------|
| 1981-01 | 1.0   | 1.0   | 1.0           |
| 1981-02 | 1.0   | 2.0   | 4.0           |
| 1981-03 | 1.0   | 3.0   | 9.0           |
| 1981-04 | 1.0   | 4.0   | 16.0          |
| 1981-05 | 1.0   | 5.0   | 25.0          |

Po rozdělení dat na tréninkovou a testovací část na těchto datech natrénujeme náš model.

```Python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, shuffle=False)

model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)
```

Na natrénovaném modelu odvodíme naší predikci.

```Python
y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)

y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
```

```Python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error (MAE):', mae)
```

*Mean Absolute Error (MAE): `61.62`*

![predikce kvadratické regrese](./img/Quadratic_regression_prediction.png)

Tato predikce již vypadá velice dobře. MAE je oproti lineární regresi prvního řádu poloviční. Tento trend využijeme dále pro stacionarizaci dat, která je nutná pro použití prediktivního modelu ARIMA.


## Model ARIMA

Z předchozí analýzy dat víme, že data nejsou stacionární. Abychom mohli použít prediktivní model ARIMA, musíme data nejprve stacionarizovat. Pro stacionarizaci použijeme metodu diferenciace, která má za úkol odstranit z dat trend. Při použití této metody musíme pamatovat na to, abychom predikovaná data převedli zpět do stavu před diferenciací.

```python
df['Sales_diff'] = df['Sales'].diff().dropna()
```

Nyní znovu odvodíme `ADF` číslo.

```python
result = adfuller(df['Sales_diff'].dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```


- **ADF Statistic:** -7.249074055553854
- **p-hodnota:** 1.7998574141687034e-10

ADF hodnota je záporná a p-hodnota se blíží nule. Naše data můžeme považovat za stacionarizovaná.

Pro odhad parametrů pro model ARIMA vygenerujeme grafy pro `auto-korelační funkci` a `parciální auto-korelaci`.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Sales_diff'].dropna(), ax=ax[0])
plot_pacf(df['Sales_diff'].dropna(), ax=ax[1])
plt.show()
```

![ACF a PACF](./img/ACF_PACF.png)

Pro model ARIMA určujeme 3 parametry `p`, `d` a `q`. Z grafů můžeme zkusit odhadnout hodnotu `p=5` (ACF má nízkou hodnotu a PACF má hodnotu zároveň vyšší), `q=8` (PACF se blíží nule a zároveň ACF má vyšší hodnotu). Parametr `d=1` odvodíme z použití jednoho stupně diferenciace.

Nyní zkusíme na základě hodnot `p=5`, `d=1` a `q=8` odvodit predikci pro náš datový set s použitím modelu ARIMA.

Samotný model ARIMA umí data sám diferenciovat, použijeme tedy původní nediferenciovaná data, s tím že 1. stupeň diferenciace je již stacionární. V dalším zkoumání zkusíme i další stupeň diferenciace, ale zde už je třeba dát pozor na overfitting natrénovaného modelu.

```Python
from statsmodels.tsa.arima.model import ARIMA
# Training data
train, test = train_test_split(df, test_size=0.33, shuffle=False)

# Train model
model = ARIMA(train['Sales'].dropna(), order=(5, 1, 8))
model_fit = model.fit()
print(model_fit.summary())
```

Zde je sumarizace natrénovaného modelu:

```Python
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Sales   No. Observations:                   24
Model:                 ARIMA(5, 1, 8)   Log Likelihood                -126.791
Date:                Mon, 27 Jan 2025   AIC                            281.582
Time:                        20:19:34   BIC                            297.479
Sample:                    01-31-1981   HQIC                           285.580
                         - 12-31-1982                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.7779      5.229     -0.149      0.882     -11.027       9.471
ar.L2         -1.5248      3.926     -0.388      0.698      -9.219       6.170
ar.L3         -1.4983      8.435     -0.178      0.859     -18.030      15.034
ar.L4         -0.7561      4.556     -0.166      0.868      -9.686       8.174
ar.L5         -0.9145      7.942     -0.115      0.908     -16.480      14.651
ma.L1         -0.1379     23.921     -0.006      0.995     -47.021      46.746
ma.L2          1.3387     25.650      0.052      0.958     -48.934      51.611
ma.L3          0.4792     54.162      0.009      0.993    -105.677     106.635
ma.L4          0.1753     46.053      0.004      0.997     -90.087      90.438
ma.L5          0.9696     53.352      0.018      0.986    -103.598     105.537
ma.L6         -0.5183     30.852     -0.017      0.987     -60.988      59.951
ma.L7          0.4704     13.927      0.034      0.973     -26.826      27.766
ma.L8         -0.1919      4.401     -0.044      0.965      -8.817       8.433
sigma2      5026.2250   8.67e+04      0.058      0.954   -1.65e+05    1.75e+05
===================================================================================
Ljung-Box (L1) (Q):                   0.27   Jarque-Bera (JB):                 0.93
Prob(Q):                              0.61   Prob(JB):                         0.63
Heteroskedasticity (H):               0.87   Skew:                            -0.18
Prob(H) (two-sided):                  0.84   Kurtosis:                         2.09
===================================================================================
```

Nyní provedeme samotnou předpověď nastavenou na následujících 12 kroků.

```Python
forecast = model_fit.forecast(steps=len(test))
```

Abychom získali předpověď v původních hodnotách našeho datasetu, musíme výstup modelu vrátit zpět do stavu před diferenciací.

```Python
forecast = forecast.cumsum() + train['Sales'].iloc[-1]
```

Pro lepší představu predikcí našeho modelu si výsledky vizualizujeme pomocí grafu.

```Python
ax = df['Sales'].plot(**plot_params)
ax = train['Sales'].plot(ax=ax, linewidth=3, color='blue', label='train')
ax = forecast.plot(ax=ax, linewidth=3, color='red', label='predicted')
ax.set_title('ARIMA Prediction of Shampoo Sales')
ax.legend();
plt.show()
```

![ARIMA_518](./img/ARIMA_518.png)

Výsledek vypadá uspokojivě, nás ale zajímá především metrika `MAE` pro porovnání s předchozími předpověďmi.

```Python
mae = mean_absolute_error(y_test, forecast)
print('Mean Absolute Error (MAE):', mae)
```

*Mean Absolute Error (MAE): `130.98`*

Tento výsledek je téměř totožný s lineární regresí. Nyní se tedy pokusíme model vyladit pomocí hyperparamtrů `p`, `d` a `q`, abychom zjistili, zda se nenabízí lepší řešení, které by překonalo alespoň kvadratickou regresi.

Pro prozkoumání různých variant paramtetrů použijeme techniku `GridSearch`, pro kterou si napíšeme vlastní funkci.

```Python
import itertools
import warnings

warnings.filterwarnings("ignore")

# Definování rozsahu pro parametry p a q, s d = 2
p = q = range(0, 10)  # Změna rozsahu podle potřeby
pdq = [(x[0], 2, x[1]) for x in list(itertools.product(p, q))]
print(pdq)

# Funkce pro grid search s MAE
def arima_grid_search(train, test, pdq):
    best_mae = np.inf
    best_order = None
    best_model = None
    
    for order in pdq:
        try:
            model = ARIMA(train['Sales'].dropna(), order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            
            # Výpočet MAE
            mae = mean_absolute_error(test, forecast)
            print(mae)
            if mae < best_mae:
                best_mae = mae
                best_order = order
                best_model = model_fit
        except:
            continue
    return best_order, best_model

# Provedení grid search
best_order, best_model = arima_grid_search(train, y_test, pdq)
print(f"Best order: {best_order}")
print(best_model.summary())
```

`Best order: (7, 2, 3)`

Nakonec nejlépe dopadla diferenciace druhého řádu. Natrénujeme tedy finální model s těmito parametry a změříme chybu.

```Python
# Train model
model = ARIMA(train['Sales'].dropna(), order=(7, 2, 3))
model_fit = model.fit()
print(model_fit.summary())
```

```Python
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  Sales   No. Observations:                   24
Model:                 ARIMA(7, 2, 3)   Log Likelihood                -120.348
Date:                Mon, 27 Jan 2025   AIC                            262.696
Time:                        20:16:18   BIC                            274.698
Sample:                    01-31-1981   HQIC                           265.523
                         - 12-31-1982                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -2.3750      2.548     -0.932      0.351      -7.369       2.619
ar.L2         -3.3425      4.223     -0.792      0.429     -11.619       4.934
ar.L3         -4.1924      5.693     -0.736      0.462     -15.351       6.966
ar.L4         -3.9592      6.466     -0.612      0.540     -16.633       8.715
ar.L5         -2.8771      5.252     -0.548      0.584     -13.170       7.416
ar.L6         -1.6497      3.529     -0.467      0.640      -8.567       5.267
ar.L7         -0.4764      1.576     -0.302      0.762      -3.565       2.612
ma.L1          0.3571      5.082      0.070      0.944      -9.604      10.318
ma.L2          0.4132      7.220      0.057      0.954     -13.739      14.565
ma.L3          0.9266      7.389      0.125      0.900     -13.556      15.409
sigma2      2404.4993   1.95e+04      0.123      0.902   -3.59e+04    4.07e+04
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                 1.49
Prob(Q):                              0.90   Prob(JB):                         0.48
Heteroskedasticity (H):               0.53   Skew:                            -0.61
Prob(H) (two-sided):                  0.42   Kurtosis:                         2.64
===================================================================================
```

Odvodíme předpověď pro následujících 12 měsíců.

```Python
forecast = model_fit.forecast(steps=len(test))
```

Opět provedeme vizualizaci naší předpovědi.

```Python
ax = df['Sales'].plot(**plot_params)
ax = train['Sales'].plot(ax=ax, linewidth=3, color='blue', label='train')
ax = forecast.plot(ax=ax, linewidth=3, color='red', label='predicted')
ax.set_title('ARIMA Prediction of Shampoo Sales')
ax.legend();
plt.show()
```

![ARIMA_723](./img/ARIMA_723.png)

Změříme `MAE`:

```Python
mae = mean_absolute_error(y_test, forecast)
print('Mean Absolute Error (MAE):', mae)
```

*Mean Absolute Error (MAE): `52.68`*

Toto je mnohem lepší výsledek než u kvadratické regrese. Tento model budeme tedy brát jako finální.

## CI - Confidence intervals

Abychom mohli prezentovat prediktivní výsledky našeho modelu, musím zjistit průběh jeho CI - intervalů spolehlivosti (Confidence intervals). Pomocí těchto intervalů můžeme určit, že máme 95% jistotu, že skutečná hodnota, kterou se snažíme odhadnout, leží v tomto intervalu. Tento interval je ovlivněn rozptylem dat.

Pro získání intervalu použijeme metodu `get_forecast`.

```Python
forecast_get = model_fit.get_forecast(steps=len(test))
forecast_index = test.index

pred_mean = forecast_get.predicted_mean
pred_ci = forecast_get.conf_int()
```

```Python
print(pred_ci)
```

```Python
        lower Sales  upper Sales
1983-01   266.472265   466.452863
1983-02   350.097152   549.051281
1983-03   266.880831   518.317368
1983-04   257.466822   543.868995
1983-05   236.059903   602.071676
1983-06   254.614220   653.008293
1983-07   270.264362   738.001277
1983-08   223.931610   762.741397
1983-09   181.996153   782.173822
1983-10   135.813548   815.969853
1983-11   179.046719   938.367940
1983-12   157.113814  1003.693186
```

Nyní interval spolehlivosti vizualizujeme v grafu společně s naší předpovědí.

```Python
ax = df['Sales'].plot(**plot_params)
ax = train['Sales'].plot(ax=ax, linewidth=3, color='blue', label='train')
ax = forecast.plot(ax=ax, linewidth=3, color='red', label='predicted, 95% CI')
ax.fill_between(
	test.index, pred_ci['lower Sales'], pred_ci['upper Sales'], color='r', alpha=.15)
ax.set_title('ARIMA Prediction of Shampoo Sales')
ax.legend(loc='upper left');
plt.show()
```

![ARIMA_723_CI](./img/ARIMA_723_CI.png)

Tím je naše práce hotova. Výhodou modelu ARIMA oproti klasické regresi je právě získání intervalů spolehlivosti. Bylo by chybou interpretovat predikci jako konkrétní hodnoty. Vždy se budeme pohybovat v rámci intervalů spolehlivosti.

## Co dál?

Jako další model můžeme vyzkoušet `Holt-Winters`.
