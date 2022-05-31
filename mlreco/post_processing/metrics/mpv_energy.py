import numpy as np
import pandas as pd
import sys, os, re

from mlreco.post_processing import post_processing
from mlreco.utils import CSVData

def mpv_energy(cfg, processor_cfg, data_blob, result, logdir, iteration):
    output = pd.DataFrame(columns=['prediction', 'truth', 'index'])
    
    index = data_blob['index']
    index = np.asarray(index)
    pred = np.vstack(result['pred'])[:, 0]
    labels = np.vstack(data_blob['label'])[:, 0]
    label_order = np.argsort(np.vstack(data_blob['label'])[:, 1])
    pred = pred[label_order]
    labels = labels[label_order]

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'mpv-energy-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):
        fout.record(('Index', 'Truth', 'Prediction'),
                    (int(event_id), labels[batch_id], pred[batch_id]))
        fout.write()

def mpv_energy_quantile(cfg, processor_cfg, data_blob, result, logdir, iteration):
    output = pd.DataFrame(columns=['p1', 'p3', 'p5', 'p7',
        'p9', 'truth', 'index'])

    labels = np.vstack(data_blob['label'])[:, 0]
    index = data_blob['index']
    index = np.asarray(index)
    pred = np.vstack(result['pred'])
    label_order = np.argsort(np.vstack(data_blob['label'])[:, 1])
    pred = pred[label_order]
    labels = labels[label_order]

    if iteration:
        append = True
    else:
        append = False

    fout = CSVData(
        os.path.join(logdir, 'mpv-energy-metrics.csv'), append=append)

    for batch_id, event_id in enumerate(index):

        probs = pred[batch_id]

        fout.record(('Index', 'Truth', 'p1', 'p3', 'p5', 'p7', 'p9'),
                    (int(event_id), labels[batch_id], 
                    probs[0], probs[1], probs[2], probs[3], probs[4]))
        fout.write()