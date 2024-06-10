# %% move files based on csv file
import numpy as np
import nibabel as nb
import os
from os import listdir, makedirs
from os.path import basename, join, dirname, isfile, isdir
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse

def compute_multi_class_dsc(gt, seg, label_unique, metric_dict):
    dsc = []
    for i in label_unique:
        gt_i = gt == i
        seg_i = seg == i
        dsc_i = compute_dice_coefficient(gt_i, seg_i)
        dsc.append(dsc_i)
        metric_dict[f'dsc_{i}'] = dsc_i
        
    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=5)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
num_workers = args.num_workers
compute_NSD = args.nsd

def compute_metrics(npz_name):
    metric_dict = {'dsc': -1.}
    if compute_NSD:
        metric_dict['nsd'] = -1.
    npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    gts = npz_gt['gts']
    segs = npz_seg['segs']
    label_unique = sorted(np.unique(gts)[1:])
    for l in range(1,8):
        metric_dict[f'dsc_{l}'] = np.nan
    if npz_name.startswith('3D'):
        spacing = npz_gt['spacing']
    
    dsc = compute_multi_class_dsc(gts, segs, label_unique, metric_dict)
    # comupute nsd
    if compute_NSD:
        if dsc > 0.2:
        # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
            if npz_name.startswith('3D'):
                nsd = compute_multi_class_nsd(gts, segs, spacing)
            else:
                spacing = [1.0, 1.0, 1.0]
                nsd = compute_multi_class_nsd(np.expand_dims(gts, -1), np.expand_dims(segs, -1), spacing)
        else:
            nsd = 0.0
    metric_dict['dsc'] = dsc
    if compute_NSD:
        metric_dict['nsd'] = nsd
        
    return npz_name, metric_dict

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['dsc'] = []
    for i in range(1,8):
        seg_metrics[f'dsc_{i}'] = []
    if compute_NSD:
        seg_metrics['nsd'] = []
    
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for i, (npz_name, metric) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                seg_metrics['case'].append(npz_name)
                # seg_metrics['dsc'].append(np.round(metric['dsc'], 4))
                # if compute_NSD:
                #     seg_metrics['nsd'].append(np.round(nsd, 4))
                for k, v in metric.items():
                    seg_metrics[k].append(np.round(v, 4))
                pbar.update()
    # for col in seg_metrics.keys():
    #     print(col, len(seg_metrics[col]))
    df = pd.DataFrame(seg_metrics)
    # rank based on case column
    df = df.sort_values(by=['case'])
    df.to_csv(csv_dir, index=False)
    print("Avg DSC: ", df['dsc'].mean(), "Avg NSD: ", df['nsd'].mean() if compute_NSD else -1.0)
    for i in df[['dsc_1', 'dsc_2', 'dsc_3', 'dsc_4', 'dsc_5', 'dsc_6', 'dsc_7']]:
        print(f"Class {i} Avg DSC: ", df[i].mean())