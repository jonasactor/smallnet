import os
import csv
import time
import json
from optparse import OptionParser

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
sys.setrecursionlimit(5000)

import math
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.signal import convolve2d

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

_globalexpectedpixel=512
_nx = 256
_ny = 256

npx = 256
hu_lb = -100
hu_ub = 400 
std_lb = 0
std_ub = 100



###
### set options
###
parser = OptionParser()
parser.add_option( "--outdir",
        action="store", dest="outdir", default="./",
        help="out location", metavar="PATH")
parser.add_option( "--savedir",
        action="store", dest="savedir", default=None,
        help="location to save images", metavar="PATH")
(options, args) = parser.parse_args()



combined_dict = { \
    'gaussian' : { \
        'dsc_l2'  : [],
        'dsc_int' : [],
        'del_l2'  : [],
        'del_int' : [],
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], }, 
    'adversarial' : { \
        'del_l2'  : [],
        'del_int' : [],
        'dsc_int' : [],
        'dsc_l2'  : [],
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], },
    'physical' : { \
        'del_int' : [] ,
        'del_l2'  : [] ,
        'dsc_int' : [] ,
        'dsc_l2'  : [] ,
        'del_l2_reg'  : [],
        'del_int_reg' : [],
        'dsc_int_reg' : [],
        'dsc_l2_reg'  : [],
        'eps_list': [], },
    }

def isolate_physical(combined_dict):
    d = []
    dreg = []
    delta = []
    deltareg = []
    for distr in combined_dict:
        d += combined_dict['physical']['dsc_l2']
        dreg += combined_dict['physical']['dsc_l2_reg']
        delta += combined_dict['physical']['del_l2']
        deltareg += combined_dict['physical']['del_l2_reg']
    print(d)
    print(dreg)
    print(delta)
    print(deltareg)


def break_by_column(single_dict):
    eps_dict = {}
    eps_dict_reg = {}
    for ie, e in enumerate(single_dict['eps_list']):
        if e not in eps_dict:
            eps_dict[e]     = [ single_dict['dsc_l2'    ][ie] ]
            eps_dict_reg[e] = [ single_dict['dsc_l2_reg'][ie] ]
        else:
            eps_dict[e]     += [ single_dict['dsc_l2'    ][ie] ]
            eps_dict_reg[e] += [ single_dict['dsc_l2_reg'][ie] ]
    return eps_dict, eps_dict_reg

def plot_changes(combined_dict):
    print(' plotting...')

    if options.savedir:

        for distr in combined_dict:
            if distr is not "physical":
                plt.figure(figsize=(5,5))
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2'],      c='r', alpha=0.5, s=9, marker='o')
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2_reg'],  c='b', alpha=0.5, s=9, marker='^')
                plt.xlabel('noise level')
                plt.ylabel('DSC score')
                plt.ylim([0.4,1.0])
                plt.tight_layout()
                plt.savefig(options.savedir+'/'+distr+'-dsc-l2.png', dpi=300)
                plt.close()

                plt.figure(figsize=(5,5))
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2'],      c='r', alpha=0.5, s=9, marker='o')
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2_reg'],  c='b', alpha=0.5, s=9, marker='^')
                plt.xlabel('noise level')
                plt.ylabel('change in DSC score due to noise')
                plt.ylim([-0.45,0.05])
                plt.tight_layout()
                plt.savefig(options.savedir+'/'+distr+'-del-l2.png', dpi=300)
                plt.close()

                plt.figure(figsize=(5,5))
                plt.scatter(combined_dict[distr]['dsc_l2'], combined_dict[distr]['dsc_l2_reg'], c=combined_dict[distr]['eps_list'], alpha=0.5, s=9)
                plt.xlabel('unregularized DSC score')
                plt.ylabel('regularized DSC score')
                plt.tight_layout()
                plt.savefig(options.savedir+'/'+distr+'-comp-l2.png', dpi=300)
                plt.close()

                plt.figure(figsize=(5,5))
                plt.scatter(combined_dict[distr]['eps_list'], [x-y for x,y in  zip(combined_dict[distr]['del_l2'], combined_dict[distr]['del_l2_reg'])], alpha=0.5, s=9)
                plt.xlabel('noise level')
                plt.ylabel('difference in regularized vs unregularized DSC score')
                plt.tight_layout()
                plt.savefig(options.savedir+'/'+distr+'-diff-l2.png', dpi=300)
                plt.close()


                ed, erd = break_by_column(combined_dict[distr])
                for e in ed:
                    print(distr, "{:.3f}".format(e), "{:.3f}".format( np.mean(ed[e])), "{:.3f}".format( np.mean(erd[e])))


#        isolate_physical(combined_dict)
#        plt.subplot(2,2,1)
#        plt.boxplot([combined_dict['physical']['dsc_l2'], combined_dict['physical']['dsc_l2_reg']])
#        plt.subplot(2,2,3)
#        plt.boxplot([combined_dict['physical']['del_l2'], combined_dict['physical']['del_l2_reg']])
#        plt.show()

    else:

        iii = 1
        nrows = len(combined_dict) -1
        for distr in combined_dict:
            if distr is not "physical":
                plt.subplot(nrows,2,iii)
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2'],      c='r', alpha=0.5, s=9, marker='o')
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['dsc_l2_reg'],  c='b', alpha=0.5, s=9, marker='^')
                iii += 1
                plt.subplot(nrows,2,iii)
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2'],      c='r', alpha=0.5, s=9, marker='o')
                plt.scatter(combined_dict[distr]['eps_list'], combined_dict[distr]['del_l2_reg'],  c='b', alpha=0.5, s=9, marker='^')
                iii += 1
        plt.show()

        isolate_physical(combined_dict)
        plt.subplot(2,2,1)
        plt.boxplot([combined_dict['physical']['dsc_l2'], combined_dict['physical']['dsc_l2_reg']])
        plt.subplot(2,2,3)
        plt.boxplot([combined_dict['physical']['del_l2'], combined_dict['physical']['del_l2_reg']])
        plt.show()

for i in range(111, 128):
    print(' reading json for scan', i)
    saved = open(options.outdir+str(i)+'/noisemaker_results.json', 'r')
    dres = json.load(saved)
    saved.close()

    for di in combined_dict:
        for listname in combined_dict[di]:
            combined_dict[di][listname] += dres[di][listname]

plot_changes(combined_dict)



