"""Plot the matched photometry and format the numerical tables for the report."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from lightpollution.adapters.storage import read_json


STYLES = {
    'lambert': ('Ламбертовская', '#325D89', (0, (5, 3))),
    'uniform_intensity': ('Равномерная сила', '#A04727', (0, (7, 2.5))),
    'half_lambert_half_uniform': ('Смесь 50:50', '#77518C', (0, (5, 2, 1, 2))),
    'off_h1': ('Два ракурса, постоянная', '#536F79', (0, (1.2, 2))),
    'off_h2': ('Два ракурса, с усилением', '#087960', '-'),
}
CITY_LABELS = {'Madrid':'Мадрид', 'Tucson':'Тусон', 'Balaguer':'Балагер'}
INSTRUMENTS = {'photopic':'DSLR', 'sqm':'SQM', 'tess':'TESS'}


def fmt(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',').replace('-', '−')


def axes_layout(count):
    fig, axes = plt.subplots(1, count, figsize=(12.6, 4.05))
    fig.subplots_adjust(left=.062, right=.978, top=.89, bottom=.30, wspace=.30)
    return fig, axes


def draw_profiles(axis, rows, column, *, median, magnitude):
    by_x = defaultdict(list)
    for row in rows:
        by_x[row[column]].append(row)
    x = sorted(by_x)
    handles = []
    for name,(label,color,linestyle) in STYLES.items():
        y = [np.median([r['prediction'][name]['prediction'] for r in by_x[p]]) for p in x]
        (handle,) = axis.plot(x, y, color=color, linestyle=linestyle, marker='.', markersize=4,
                              lw=2.15, label=label)
        handles.append(handle)
    if median:
        observed = [np.median([r['value'] for r in by_x[p]]) for p in x]
        axis.scatter([r[column] for r in rows], [r['value'] for r in rows], s=13, color='#536B76', alpha=.45)
    else:
        observed = [by_x[p][0]['value'] for p in x]
    handle = axis.scatter(x,observed,s=33,edgecolor='white',linewidth=.6,color='#122F40',zorder=7,label='Замер')
    if magnitude:
        axis.invert_yaxis()
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'.replace('.',',')))
    return [handle,*handles]


def finish(fig, handles, path):
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52,.035), ncol=3,
               fontsize=10.5, handlelength=2.4)
    fig.savefig(path,dpi=220,facecolor='white')
    plt.close(fig)


def plot_all(data, directory):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
        'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'axes.axisbelow':True,'grid.color':'#DEE7EA','legend.frameon':False})
    madrid = [r for r in data['rows'] if r['city']=='Madrid']
    fig, axes = axes_layout(3)
    date_labels = ('19–20.02.2012','03.12.2013','28–29.12.2013')
    for ax,night,label in zip(axes,sorted({r['night'] for r in madrid}),date_labels,strict=True):
        handles = draw_profiles(ax,[r for r in madrid if r['night']==night], 'distance_km',median=False,magnitude=True)
        ax.set(title=label,xlabel='Расстояние, км',ylabel='SQM, mag/arcsec²')
    finish(fig,handles,directory/'madrid.png')

    tucson = [r for r in data['rows'] if r['city']=='Tucson']
    fig, axes = axes_layout(3)
    fig.subplots_adjust(wspace=.32)
    camera = [r for r in tucson if r['band']=='photopic']
    nights = sorted(r['night'] for r in camera)
    draw_profiles(axes[0],[r | {'date_index':nights.index(r['night'])} for r in camera], 'date_index',median=False,magnitude=False)
    axes[0].set(title='Камера, 5,5 км',ylabel='Яркость, мкд/м²', xlabel='Дата UTC в 2019 году',
                xticks=range(len(nights)),xticklabels=[d[8:]+'.'+d[5:7] for d in nights])
    for ax,band in zip(axes[1:],('sqm','tess'),strict=True):
        handles = draw_profiles(ax,[r for r in tucson if r['band']==band], 'distance_km',median=True,magnitude=True)
        ax.set(title=INSTRUMENTS[band], xlabel='Расстояние, км',ylabel='mag/arcsec²')
    finish(fig,handles,directory/'tucson.png')

    balaguer = sorted([r for r in data['rows'] if r['city']=='Balaguer'],key=lambda r:r['distance_km'])
    fig, (left,right) = axes_layout(2)
    handles = draw_profiles(left,balaguer,'distance_km',median=False,magnitude=False)
    left.set(title='Зенитная яркость', xlabel='Расстояние, км', ylabel='Яркость, мкд/м²',yscale='log')
    right.axhline(50,color='#8495A0',lw=1,linestyle='--')
    for name in ('lambert','off_h1','off_h2'):
        label,color,linestyle = STYLES[name]
        right.plot([r['distance_km'] for r in balaguer],
                   [r['prediction'][name]['city_fraction']*100 for r in balaguer],
                   color=color,linestyle=linestyle,marker='.',lw=2.15)
    right.set(title='Доля Балагера в искусственном свете', xlabel='Расстояние, км', ylabel='Доля города, %',ylim=(0,102))
    finish(fig,handles,directory/'balaguer.png')


def point_table(rows, tucson=False):
    names = 'Дата UTC | Прибор | км' if tucson else 'км'
    header = f'| {names} | Замер | Ламберт | Равн. | 50:50 | Пост. | Усил. | Доля города |'
    sep = '|' + '---|'*(10 if tucson else 8)
    output = [header,sep]
    for row in rows:
        digits = 3 if row['band']=='photopic' else 2
        pred = row['prediction']
        values = [fmt(row['distance_km'],1),fmt(row['value'],digits)]
        values += [fmt(pred[name]['prediction'],digits) for name in STYLES]
        values.append(fmt(pred['off_h2']['city_fraction']*100,0)+'%')
        if tucson:
            night = row['night']
            values = [night[8:]+'.'+night[5:7], INSTRUMENTS[row['band']]]+values
        output.append('| '+' | '.join(values)+' |')
    return '\n'.join(output)


def tables(data):
    berlin = read_json(Path(__file__).resolve().parents[1]/'artifacts'/'berlin-propagation-20261126'/'results.json')
    lines = ['| Город | Замеры | Ламберт | Равн. | 50:50 | Пост. | Усил. |','|---|---:|---:|---:|---:|---:|---:|']
    old = berlin['metrics']['all']['metrics']['aeronet']['equal_dnb']
    lines.append('| Берлин | 12 | '+' | '.join(fmt(old[name]['mean_absolute_pct'])+'%' for name in STYLES)+' |')
    for city,label in CITY_LABELS.items():
        stats = data['metrics'][city]['all']
        lines.append('| '+label+' | '+str(stats['n'])+' | '+' | '.join(fmt(stats['cases'][n]['mae_pct'])+'%' for n in STYLES)+' |')
    for city,label in CITY_LABELS.items():
        groups = data['metrics'][city]
        lines += ['', '### '+label, '',
                  '| Разрез | Замеры | Ламберт | Равн. | 50:50 | Пост. | Усил. |','|---|---:|---:|---:|---:|---:|---:|']
        source = [('Преобладает город',groups['city_dominant'])]
        source += (list(groups['nights'].items()) if city=='Madrid' else [(INSTRUMENTS[k],v) for k,v in groups['bands'].items()])
        for description, row in source:
            if row:
                lines.append('| '+description+' | '+str(row['n'])+' | '+' | '.join(fmt(row['cases'][k]['mae_pct'])+'%' for k in STYLES)+' |')
        selected = [r for r in data['rows'] if r['city']==city]
        for night in sorted({r['night'] for r in selected}) if city=='Madrid' else (None,):
            subset=[r for r in selected if night is None or r['night']==night]
            rows=sorted(subset,key=lambda r:(r['night'],r['distance_km'],r['band']))
            if night is not None:
                lines += ['', '### '+night]
            lines += ['',point_table(rows,tucson=city=='Tucson')]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('figures',type=Path)
    parser.add_argument('tables',type=Path)
    args = parser.parse_args()
    data = read_json(args.results)
    args.figures.mkdir(parents=True,exist_ok=True)
    plot_all(data,args.figures)
    args.tables.write_text(tables(data),encoding='utf-8')


if __name__=='__main__':
    main()
