"""Redraw the four public charts from aggregate measurements only.

Run: python docs/plot_benchmarks.py; requires NumPy and Matplotlib.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def short(s):return s.removesuffix('.hgr').removesuffix('.mtx').removeprefix('ISPD98_').replace('ibm','IBM')


def draw(path,output):
    data=json.loads(path.read_text());output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.spines.top':False,'axes.spines.right':False})
    for timing in [False,True]:
        rows=data['qubo'];fig,axes=plt.subplots(1,3,figsize=(10.5,10.4),sharey=True)
        y=np.arange(len(rows))
        for panel,ax in enumerate(axes):
            f=np.array([r['settings'][panel]['fem']['seconds' if timing else 'cut'] for r in rows])
            k=np.array([r['settings'][panel]['kahypar']['seconds' if timing else 'cut'] for r in rows]);ratio=f/k
            ax.barh(y+.16,np.ones(len(rows)),height=.29,color='#d2d8df',label='KaHyPar')
            ax.barh(y-.16,ratio,height=.29,color='#257da4',label='QUBO FEM')
            ax.axvline(1,color='#5b6670',linewidth=.6,linestyle='--');ax.grid(axis='x',alpha=.13)
            for i,row in enumerate(rows):
                cell=row['settings'][panel]
                for dy,method,width in [(-.16,'fem',ratio[i]),(.16,'kahypar',1.)]:
                    entry=cell[method]
                    label=f"{entry['seconds']:.2f}s" if timing else f"{entry['cut']:,.0f}; {entry['imbalance_percent']:.3f}%"
                    ax.text(width+(.07 if not timing else width*.08),i+dy,label,va='center',fontsize=5.8)
            if timing:
                ax.set_xscale('log');ax.set_xlim(.1,max(80,ratio.max()*2.8));ax.set_xticks([.1,1,10,100]);ax.set_xticklabels(['0.1','1','10','100'])
            else:ax.set_xlim(0,max(2.2,ratio.max()+1.5))
            ax.set_xlabel(('Charged time / KaHyPar time' if timing else 'Native cut / KaHyPar cut')+f"\nAllowed imbalance {rows[0]['settings'][panel]['allowed_imbalance_percent']}%")
            ax.set_yticks(y);ax.text(.5,-.075,f'({chr(97+panel)})',transform=ax.transAxes,ha='center',fontsize=10)
        axes[0].set_yticklabels([short(r['instance']) for r in rows],fontstyle='italic');axes[0].invert_yaxis()
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
        fig.subplots_adjust(left=.115,right=.98,top=.965,bottom=.09,wspace=.12)
        stem='qubo_runtime' if timing else 'qubo_quality';fig.savefig(output/(stem+'.png'),dpi=180);plt.close(fig)
    colors=['#8d9daa','#287ba0','#ce8648','#cbd1d8'];labels=['Off','Shared','Mixed','KaHyPar']
    for timing in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(10.5,3.15))
        for panel,(ax,row) in enumerate(zip(axes,data['hubo'])):
            values=[r['seconds' if timing else 'cut'] for r in row['arms']]
            ax.bar(np.arange(4),values,color=colors,width=.7)
            if not timing:ax.set_yscale('log');ax.set_ylim(min(values)*.6,max(values)*2)
            else:ax.set_ylim(0,max(values)*1.24)
            for i,(value,arm) in enumerate(zip(values,row['arms'])):
                text=f'{value:.2f}s' if timing else f"{value:,.0f}\n{arm['imbalance_percent']:.3f}%"
                ax.text(i,value*(1.09 if not timing else 1)+(.025*max(values) if timing else 0),text,ha='center',va='bottom',fontsize=6.5)
            ax.set_xticks(np.arange(4),labels,fontsize=7);ax.set_ylabel('Charged time (s)' if timing else 'Native cut (log scale)')
            ax.text(.97,.97,short(row['instance']),transform=ax.transAxes,ha='right',va='top',fontstyle='italic')
            ax.text(.5,-.22,f'({chr(97+panel)})',transform=ax.transAxes,ha='center',fontsize=10);ax.grid(axis='y',alpha=.13)
        fig.subplots_adjust(left=.075,right=.99,top=.96,bottom=.22,wspace=.28)
        stem='hubo_runtime' if timing else 'hubo_quality';fig.savefig(output/(stem+'.png'),dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path(__file__).parent/'benchmarks/paper_tables.json')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'images')
    a=p.parse_args();draw(a.data,a.output)
