# -*- coding: utf-8 -*-
"""
From: https://www.esrl.noaa.gov/psd/enso/mei/
If we use percentiles (say, the lower and upper 30%iles) to define La Nina and
El Nino, respectively, MEI ranks from 1-21 denote strong to weak La Nina
conditions, while 48-68 (49-69) denote weak to strong El Nino conditions.
If one uses the quintile definition for (moderate or stronger) ENSO events, MEI
ranks from 1-14 would denote La Nina, while 55-68 (56-69)  would denote El Nino.
Finally, the comparison figures on this website refer to strong ENSO events,
such as might be defined by the top 7 (upper decile) ranks, such as 1-7 for
La Nina, and 62-68 (63-69) for El Nino.
"""
## IMPORT Packages #############################################################
import numpy as np
import pandas as pd
from scipy import stats

# MatplotLib libraries
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
#np.seterr(divide='ignore', invalid='ignore')
################################################################################
def calc_year(valores):
    """
    A partir de un array de strings de archivos de rendimientos
    de MAGYP crea un array de enteros con los años
    """
    str_aux = valores[0]
    year_i = int(str_aux[0:4]) + 1
    str_aux = valores[-1]
    year_f = int(str_aux[0:4]) + 1
    years = np.arange(year_i, year_f + 1)
    return years


def calc_r(x, y, coeffs):
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    r_squared = ssreg / sstot
    return r_squared


def grafico_dispersion(x, y, **kwargs):
    """
    Ajustamos un grafico de dispersión y colocamos la regresion hecha
    ya sea polinomio, lineal o promedio.
    Solo para hacer una revision.
    """
    os.makedirs('./tmp/', exist_ok=True)
    if kwargs['mode'] == 'poli':
        p = np.poly1d(kwargs['cf'])
        yhat = p(x)
        ley = r'$R^2$ = ' + str(kwargs['r2'])
    elif kwargs['mode'] == 'lineal':
        yhat = kwargs['m']*x + kwargs['n']
        ley = r'$R^2$ = ' + str(kwargs['r2'])
    else:
        yhat = kwargs['ave']*np.ones(len(x))
        ley = 'Promedio'

    my_dpi = 200
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='white', dpi=my_dpi)
    ax.plot(x, y, color='#ed2026', ls = '--', marker='s', ms=5.,
            linewidth=0.5, label='Rinde ' + kwargs['Cultivo'] + ' (Kg/Ha)')
    ax.plot(x, yhat, color='#3937e5', ls=':', linewidth=1., label=ley)
    ax.set_title(kwargs['title'], fontsize=15)
    lgnd = ax.legend(loc='best', prop={'size': 9})
    plt.savefig('./tmp/' + kwargs['figname'], bbox_inches='tight')
    # Close figure
    plt.close()


################################################################################
# Datos Iniciales #
folder = 'c:/Felix/ORA/python_scripts/Test_GITHUB/MEI/'
arc = 'MEI_values.txt'
cultivo = 'Trigo'  # Puede ser: Trigo, Maiz, Algodon, Girasol, Soja1, Soja2

# Codigos de calculo
MEI = pd.read_csv(folder + 'datos/' + arc)  # Leemos archivo de MEI
rend = pd.read_csv(folder + 'datos/' + cultivo + '.csv',\
                   sep=';', thousands='.')  # Rendimientos

rank = MEI.rank()
rank['YEAR'] = MEI['YEAR']
# Renombramos las columnas del archivo
rend.rename(columns={'Cultivo':'CLT', 'Campagna': 'C', 'Provincia':'Prov',\
                     'Departamento': 'Dpto', 'Sup. Sembrada':'SS',\
                     'Sup. Cosechada':'SC', 'Produccion':'P',\
                     'Rendimiento':'R0'}, inplace=True)

dptos = rend.Dpto.unique()  # A partir de estos vamos a iterar
prov  = rend.Prov.unique()

for pv in prov[0:1]:
    for dp in dptos[0:1]:
        tabla = rend[np.logical_and(rend['Dpto'] == dp, rend['Prov'] == pv)]
        r_est = 1000*(tabla['P'].values/tabla['SS'].values)  # Asi queda en Kg/Ha
        tabla = tabla.assign(R=r_est)
        years = calc_year(tabla['C'].values)
        tabla = tabla.assign(Yr=years)
        # Ajustamos un Polinomio de orden 2
        cf = np.polyfit(years, r_est, 2)
        r2 = calc_r(years, r_est, cf)
        titulo = cultivo + ' en ' + tabla['Dpto'].values[0] + ', ' + tabla['Prov'].values[0]
        figname = cultivo + '_' + tabla['Dpto'].values[0] + '_' + tabla['Prov'].values[0] +'_poli.jpg'
        kwargs = {'mode': 'poli', 'cf':cf, 'r2':r2, 'Cultivo':cultivo,\
                  'title':titulo,'figname':figname }
        grafico_dispersion(years, r_est, **kwargs)
        # Ajustamos una regresión lineal
        m, n, rv, pv, std_err = stats.linregress(years, r_est)
        titulo = cultivo + ' en ' + tabla['Dpto'].values[0] + ', ' + tabla['Prov'].values[0]
        figname = cultivo + '_' + tabla['Dpto'].values[0] + '_' + tabla['Prov'].values[0] +'_lineal.jpg'
        kwargs = {'mode': 'lineal', 'm':m, 'n':n, 'r2':r2, 'Cultivo':cultivo,\
                  'title':titulo,'figname':figname }
        grafico_dispersion(years, r_est, **kwargs)
        # Ajustamos un Polinomio de orden 0 --> Promedio
