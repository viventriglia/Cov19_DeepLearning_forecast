import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def progressbar(iterable, prefix="", size=50, file=sys.stdout):
    """
    Show a progress bar for loop calculations

    Keywords:
    iterable --> iterable for which the bar has to be displayed
    prefix --> any text you wish to write before the bar
    size --> bar width
    file --> file objects for standard output
    """
    count = len(iterable)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i%%\r" % (prefix, "#"*x, "."*(size-x), int(100*j/count)))
        file.flush()        
    show(0)
    for i, item in enumerate(iterable):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

def data_by_region(region_name):
    """
    Return a DataFrame with up-to-date figures from Protezione Civile's GitHub

    Keyword:
    region_name --> str, name of the region
    """
    # Feed web-data into pandas, down-selecting a region and formatting the date
    ts = pd.read_csv(
        'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    regione = ts[ts['denominazione_regione'] == region_name.title()].drop(columns=['stato', 'codice_regione', 'lat', 'long'])
    regione['data'] = pd.to_datetime(regione.data).dt.strftime('%d-%m-%y')

    # Add columns for daily fatalities, swabs and positivity rate
    incremento_deceduti = np.diff(regione.deceduti)
    regione['incremento_deceduti'] = np.insert(incremento_deceduti, 1, 0)  # insert a 0 for the first day
    incremento_tamponi_mol = np.diff(regione.tamponi_test_molecolare)
    regione['incremento_tamponi_molecolari'] = np.insert(incremento_tamponi_mol, 1, 0)
    regione['tasso_positività'] = regione['nuovi_positivi'] / regione['incremento_tamponi_molecolari']

    return regione

def plot_smooth(region, type=None, time_window=21, smooth_window=7, save=False):
    """
    Plot different kind of data (daily infections, deaths or ICU admissions), together with a forward-backward EMA to filter fluctuations out

    Keywords:
    region --> DataFrame, output from data_by_region()
    type --> str; 'positivi' (new infections), 'deceduti' (deaths), 'TI' (ICU admissions) or 'positività' (positivity rate); defaults to 'positivi'
    time_window --> int, number of days to plot to date; default, 21 (3 weeks)
    smooth_window --> int, number of days used for smoothing; default, 7 (1 week)
    save --> bool, choose whether to save the displayed image; default, False
    """
    time_window, smooth_window = int(time_window), int(smooth_window)
    plt.figure(figsize=(12, 6))

    if type == 'positivi' or type == None:
        type, column = 'nuovi positivi', 'nuovi_positivi'
        if region.denominazione_regione.iloc[0] == 'Campania':
            # November '20 (http://dati.istat.it/Index.aspx?DataSetCode=DCIS_POPORESBIL1)
            pop_campania = 5687845
            # cases / 100k people (over the past 7 days) -- the "red zone" threshold is 250
            incidenza = int(np.around(10 ** 5 * region[column].tail(7).sum() / pop_campania, 0))
            plt.plot([], [], ' ', label=f'{incidenza} casi / 100mila abitanti')
    elif type == 'deceduti':
        type, column = 'incremento deceduti', 'incremento_deceduti'
    elif type == 'TI':
        type, column = 'ingressi in TI', 'ingressi_terapia_intensiva'
    elif type == 'positività':
        type, column = 'tasso di positività', 'tasso_positività'
        monthly_rolling_median = region.tasso_positività.rolling(30).median().values[-1]
        weekly_rolling_median = region.tasso_positività.rolling(7).median().values[-1]
        plt.plot([], [], ' ', label=f'rolling median (30 giorni): {np.round(100*monthly_rolling_median,1)}%')
        plt.plot([], [], ' ', label=f'rolling median (7 giorni): {np.round(100*weekly_rolling_median,1)}%')
    else:
        print("I'm sorry, I don't understand; remember that the allowed types are: positivi, deceduti, TI and positività.")

    # Forward-backward EMA
    f_ema = region[column].ewm(span=smooth_window).mean()
    fb_ema = f_ema[::-1].ewm(span=smooth_window).mean()[::-1]

    # Plot
    plt.plot(region.data.tail(time_window), region[column].tail(time_window),
             marker='o', ms=5, ls=':', lw=0.8, c='tab:red')
    plt.plot(region.data.tail(time_window), fb_ema.tail(time_window),
             lw=1.5, c='k', label=f'smoothing esponenziale ({smooth_window} giorni)')
    tick_freq = int(round(time_window/10,0))
    plt.xticks(plt.xticks()[0][::tick_freq])  # make ticks less dense
    ylabel = [word.title() if len(word)>2 else word for word in type.split()]
    plt.ylabel(f"{' '.join(ylabel)}")
    plt.legend(frameon=False)
    plt.title(f'Regione {region.denominazione_regione.iloc[0]} (ultimi {time_window} giorni)')

    if save:
        plt.savefig(f'{type}_{region.denominazione_regione.iloc[0]}_tw={time_window}_sw={smooth_window}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def mad(arr):
    """Median Absolute Deviation: a robust version of standard deviation"""
    arr = np.ma.array(arr).compressed()     # a (compressed) masked array is a quick way to avoid NaNs
    med = np.median(arr)
    return np.median(np.abs(arr - med))