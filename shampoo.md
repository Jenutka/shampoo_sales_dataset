# Predikce časových řad - Shampoo Sales dataset

![Shampoo Sales dataset](./img/shampoo.jpg)

## Obsah

- [Úvod do problematiky](#úvod-do-problematiky)
- [Použité nástroje](#použité-nástroje)
- [Představení datasetu](#představení-datasetu)
- [Načtení a transformace dat](#načtení-a-transformace-dat)
- [Analýza dat](#analýza-dat)
- [Diferenciace dat](#diferenciace-dat)


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

Nyní si pomocí modulu `Matplotlib` vygenerujeme první vizualizaci dat, která nám poskytne daleko lepší představu o tom, jak data celkově vypadají. Zároveň si pomocí funkce `regplot()` proložíme trendovou linii dat do grafu (modře). V grafu vidíme, že trend je vzestupný a vcelku dobře vystihuje okolní data.

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


## Diferenciace dat

Na základě výsledků z analýzy dat se tedy musíme pokusit odstranit z dat šum a  data stacionarizovat, abychom měli dobrý základ pro prediktivní model.