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
    years = np.asarray([float(yr_a[0:4]) for yr_a in valores])
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


def grafico_stemporal(x, y, **kwargs):
    """
    Ajustamos un grafico de dispersión y colocamos la regresion hecha
    ya sea polinomio, lineal o promedio.
    Solo para hacer una revision.
    """
    os.makedirs('../tmp/', exist_ok=True)
    if kwargs['mode'] == 'poli':
        p = np.poly1d(kwargs['cf'])
        yhat = p(x)
        ley = r'$R^2$ = %(r2)3.3f'%{'r2':kwargs['r2']}
    elif kwargs['mode'] == 'lineal':
        yhat = kwargs['m']*x + kwargs['n']
        ley = r'$R^2$ = %(r2)3.3f'%{'r2':kwargs['r2']}
    else:
        yhat = kwargs['ave']*np.ones(len(x))
        ley = 'Promedio'

    my_dpi = 200
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='white', dpi=my_dpi)
    ax.plot(x, y, color='#ed2026', ls = '--', marker='s', ms=5.,
            linewidth=0.5, label='Rinde (Kg/Ha)')
    ax.plot(x, yhat, color='#3937e5', ls=':', linewidth=1., label=ley)
    ax.set_title(kwargs['title'], fontsize=15)
    ax.set_xlim([x[0]-1, x[-1] + 1])
    ax.set_xticks(x[0:-1:5])
    ax.set_xticklabels(kwargs['xlabel'][0:-1:5], fontsize=8)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.4)
    lgnd = ax.legend(loc='best', prop={'size': 9})
    plt.savefig('../tmp/' + kwargs['figname'], bbox_inches='tight')
    # Close figure
    plt.close()


def define_criterio(x):
    """
    Funcion para ver si cumple el criterio de calculo de apartamientos
    a partir de la tabla de datos
    """
    if len(x) > 1:
        nyr = len(x)
        difes = np.max(np.diff(x))
        criterio = np.logical_and(nyr > 20, difes < 5)
    else:
        nyr = 1
        difes = np.nan
        criterio = False
    return criterio, nyr, difes


def grafico_apartamiento(x, y, **kwargs):
    """
    Grafico de apartamientos
    """
    my_dpi = 200
    fig, ax = plt.subplots(nrows=1, ncols=1, facecolor='white', dpi=my_dpi)
    if 'enso' in kwargs:
        inino = kwargs['enso'] == 1
        inina = kwargs['enso'] == -1
        ineutro = kwargs['enso'] == 0

        ax.plot(x[inino], y[inino]*100., color='#ed2026', ls = 'None',\
                marker='s', ms=6., label=u'AP- El Niño')
        ax.plot(x[inina], y[inina]*100., color='#459def', ls = 'None',\
                marker='s', ms=6., label=u'AP- La Niña')
        ax.plot(x[ineutro], y[ineutro]*100., color='#b0b4b7', ls = 'None',\
                marker='s', ms=6., label=u'AP- Neutro')
        ax.plot(x, y*100., color='black', ls = '--',
                linewidth=0.6, label='_nolegend_')
        ax.plot(x, 100.*np.nanstd(y)*np.ones(len(x)), color='#bfbfbf', lw=0.5,\
                label=r'$\pm \sigma$ =%(sig)3.2f'%{'sig':100.*np.nanstd(y)})
        ax.axhline(100*np.nanstd(y), color='#bfbfbf', lw=0.8)
        ax.axhline(-100*np.nanstd(y), color='#bfbfbf', lw=0.8)
    else:
        ax.plot(x, y*100., color='#ed2026', ls = '--', marker='s', ms=5.,
                linewidth=0.5, label='Apartamiento porcentual')
    # Accesorios del grafico
    ax.set_title(kwargs['title'], fontsize=15)
    # Eje X
    ax.axhline(0, color='black', lw=1.1)
    ax.set_xticks(x[0:-1:5])
    ax.set_xticklabels(kwargs['xlabel'][0:-1:5], fontsize=8)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.4)
    lgnd = ax.legend(loc='best', prop={'size': 9})
    # Guardar la figura
    plt.savefig('../tmp/' + kwargs['figname'], bbox_inches='tight')
    plt.close()


def explore_MEI(df):
    """
    Clasificacion Periodo
        +1 ---> El Nino
        0  ---> Neutral
        -1 ---> La Nina
    """
    cond = df['YEAR'] >= 1980
    x = df['YEAR'][cond].values
    rnk = df[['SEPOCT', 'OCTNOV', 'NOVDEC']][cond].values
    # y_flat = np.ndarray.flatten(y)
    clas = np.zeros(len(x))
    for ind, yr in enumerate(x):
        vals = rnk[ind]
        nino_clas = np.sum(np.logical_and(vals>=48, vals<=68))
        nina_clas = np.sum(np.logical_and(vals>=1, vals<=21))
        print(nina_clas)
        if nino_clas >= 3:
            clas[ind] = 1
        elif nina_clas >=3:
            clas[ind] = -1
        else:
            clas[ind] = 0
    return x, clas


def clas_rend_by_enso(enso, AP):
    """
    Genera DataFrame de Pandas con la tabla de clasificacion
    segun enso y rendimiento
    Se utiliza un valor de una desviacion estandar para decir si son altos,
    bajos o normales
    """
    std_val = np.nanstd(AP)
    fnino = 100./np.sum(enso == 1)
    fnina = 100./np.sum(enso == -1)
    fneutro = 100./np.sum(enso == 0)
    tabla = pd.DataFrame(columns=['NINO', 'NINA', 'NEUTRO', 'STD'],
                         index=['Altos', 'Norm','Bajos', 'Total'])
    vals = [fnino*np.sum(AP[enso == 1] >= std_val),\
            fnina*np.sum(AP[enso == -1] >= std_val),\
            fneutro*np.sum(AP[enso == 0] >= std_val),\
            std_val]
    tabla.loc['Altos'] = vals
    vals = [fnino*np.sum(np.logical_and(AP[enso == 1] <= std_val,\
                                        AP[enso == 1] >= -std_val)),\
            fnina*np.sum(np.logical_and(AP[enso == -1] <= std_val,\
                                        AP[enso == -1] >= -std_val)),\
            fneutro*np.sum(np.logical_and(AP[enso == 0] <= std_val,\
                                          AP[enso == 0] >= -std_val)),\
            np.nan ]
    tabla.loc['Norm'] = vals
    vals = [fnino*np.sum(AP[enso == 1] <= -std_val),\
            fnina*np.sum(AP[enso == -1] <= -std_val),\
            fneutro*np.sum(AP[enso == 0] <= -std_val),\
             np.nan]
    tabla.loc['Bajos'] = vals
    vals = [np.sum(enso == 1),\
            np.sum(enso == -1),\
            np.sum(enso == 0), np.nan]
    tabla.loc['Total'] = vals
    return tabla
################################################################################
# Datos Iniciales #
folder = 'c:/Felix/ORA/python_scripts/Test_GITHUB/MEI/'
arc = 'MEI_values.txt'
cultivo = 'Trigo'  # Puede ser: Trigo, Maiz, Algodon, Girasol, Soja1, Soja2

# Codigos de calculo
f = open('../tmp/logfile.txt', 'w')
f.write('Este es el archivo de log para calculo de rendimientos vs ENSO\n')
# Calculos con el MEI
MEI = pd.read_csv(folder + 'datos/' + arc)  # Leemos archivo de MEI
rend = pd.read_csv(folder + 'datos/' + cultivo + '.csv',\
                   sep=';', thousands='.')  # Rendimientos
rank = MEI.rank()
rank['YEAR'] = MEI['YEAR']
enso_yr, enso_clas = explore_MEI(rank)
# Renombramos las columnas del archivo
rend.rename(columns={'Cultivo':'CLT', 'Campagna': 'C', 'Provincia':'Prov',\
                     'Departamento': 'Dpto', 'Sup. Sembrada':'SS',\
                     'Sup. Cosechada':'SC', 'Produccion':'P',\
                     'Rendimiento':'R0'}, inplace=True)

dptos = rend.Dpto.unique()  # A partir de estos vamos a iterar
prov  = rend.Prov.unique()
for pv in prov:
    for dp in dptos:
        print(dp + ', ' + pv)
        f.write('Trabajamos en ' + dp + ', ' + pv + ' \n')
        tabla = rend[np.logical_and(rend['Dpto'] == dp, rend['Prov'] == pv)]
        years = calc_year(tabla['C'].values)
        criterio, nyr, difes = define_criterio(years)
        if criterio:
            enso = enso_clas[np.isin(enso_yr, years)]
            tabla = tabla.assign(Yr=years)
            tabla = tabla.assign(ENSO=enso)
            r_est = 1000*(tabla['P'].values/tabla['SS'].values)  # Asi queda en Kg/Ha
            tabla = tabla.assign(R=r_est)
            # ###############################
            # Ajustamos un Polinomio de orden 2
            cf = np.polyfit(years, r_est, 2)
            r2 = calc_r(years, r_est, cf)
            p = np.poly1d(cf)
            r_hat = p(years)
            tabla = tabla.assign(RT=r_hat)
            titulo = cultivo + ' en ' + tabla['Dpto'].values[0] + ', ' + tabla['Prov'].values[0]
            figname = cultivo + '_' + tabla['Dpto'].values[0] + '_' + tabla['Prov'].values[0] +'_poli.jpg'
            kwargs = {'mode': 'poli', 'cf':cf, 'r2':r2, 'Cultivo':cultivo,\
                      'title':titulo,'figname':figname, 'xlabel':tabla['C'] }
            grafico_stemporal(years, r_est, **kwargs)
            # ###############################
            # Ajustamos una regresión lineal
            m, n, rv, pval, std_err = stats.linregress(years, r_est)
            titulo = cultivo + ' en ' + tabla['Dpto'].values[0] + ', ' + tabla['Prov'].values[0]
            figname = cultivo + '_' + tabla['Dpto'].values[0] + '_' + tabla['Prov'].values[0] +'_lineal.jpg'
            kwargs = {'mode': 'lineal', 'm':m, 'n':n, 'r2':rv*rv, 'Cultivo':cultivo,\
                      'title':titulo,'figname':figname, 'xlabel':tabla['C'] }
            grafico_stemporal(years, r_est, **kwargs)
            # ###############################
            # Calculamos apartamientos con respecto a la tendencia
            DIF = r_est - r_hat
            AP = DIF/r_hat
            tabla = tabla.assign(AP=AP)
            titulo = cultivo + ' en ' + tabla['Dpto'].values[0] + ', ' + tabla['Prov'].values[0]
            figname = cultivo + '_' + tabla['Dpto'].values[0] + '_' + tabla['Prov'].values[0] +'_apartamiento.jpg'
            kwargs = {'Cultivo':cultivo,'title':titulo,'figname':figname,\
                      'xlabel':tabla['C'], 'enso':enso}
            grafico_apartamiento(years, AP, **kwargs)
            # ###############################
            # Clasificamos las campagnas segun +-1 STD de Apartamientos

            ntabla = clas_rend_by_enso(enso, AP)

            # #########################################################
            nombre = '../tmp/' + cultivo + '_' + tabla['Dpto'].values[0] +\
                     '_' + tabla['Prov'].values[0] + '.xlsx'
            writer = pd.ExcelWriter(nombre, engine = 'xlsxwriter')
            tabla.to_excel(writer, sheet_name = 'Datos_x_Campagna')
            ntabla.to_excel(writer, sheet_name = 'Estadisticos')
            writer.save()
            writer.close()
            del(tabla)
        else:
            txt_sin = dp + ', ' + pv + ' NO cumple criterios'
            print('--------------')
            print(txt_sin)
            f.write('--------------')
            f.write(txt_sin + ' \n')
            f.write('--------------')
            print('--------------')
f.close()
