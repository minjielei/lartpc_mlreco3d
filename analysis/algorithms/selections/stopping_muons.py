from collections import OrderedDict
from turtle import update
from sklearn.decomposition import PCA

from analysis.classes.ui import FullChainEvaluator, FullChainPredictor
from analysis.decorator import evaluate
from mlreco.utils.gnn.evaluation import clustering_metrics
from mlreco.utils.gnn.cluster import get_cluster_label

from pprint import pprint
import time
import numpy as np
from scipy.spatial.distance import cdist


@evaluate(['stopping_muons_cells', 'stopping_muons_pred', 'stopping_muons_true'], mode='per_batch')
def stopping_muons(data_blob, res, data_idx, analysis_cfg, cfg):
    """
    Selection of stopping muons
    ===========================

    To convert dQ/dx from ADC/cm to MeV/cm. We want a sample as pure
    as possible, hence the option to enforce the presence of a Michel
    electron at the end of the muon.

    Configuration
    =============

    """
    muon_cells, muons, true_muons = [], [], []

    deghosting          = analysis_cfg['analysis']['deghosting']
    processor_cfg       = analysis_cfg['analysis'].get('processor_cfg', {})

    spatial_size        = processor_cfg['spatial_size']
    #selection_threshold = processor_cfg['selection_threshold']
    bin_size            = processor_cfg['bin_size']
    # Whether we are running on MC or data
    data                = processor_cfg.get('data', False)
    # Whether to restrict to tracks that are close to Michel voxels
    # threshold =-1 to disable, otherwise it is the threshold below which we consider the track
    # might be attached to a Michel electron.
    Michel_threshold    = processor_cfg.get('Michel_threshold', -1)
    # Whether to enforce PID constraint (predicted as muon only)
    pid_constraint      = processor_cfg.get('pid_constraint', False)
    # Avoid hardcoding labels
    muon_label       = processor_cfg.get('muon_label', 2)
    track_label      = processor_cfg.get('track_label', 1)

    # Initialize analysis differently depending on data/MC setting
    if not data:
        predictor = FullChainEvaluator(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)
    else:
        predictor = FullChainPredictor(data_blob, res, cfg, analysis_cfg, deghosting=deghosting)

    image_idxs = data_blob['index']

    # TODO check that 0 is actually drift direction
    # TODO check that 2 is actually vertical direction
    x, y, z = 0, 1, 2

    pca = PCA(n_components=2)

    for i, index in enumerate(image_idxs):
        pred_particles = predictor.get_particles(i, only_primaries=False)

        # Match with true particles if available
        if not data:
            true_particles = predictor.get_true_particles(i, only_primaries=False)
            # Match true particles to predicted particles
            true_ids = np.array([p.id for p in true_particles])
            matched_particles = predictor.match_particles(i, mode='true_to_pred', min_overlap=0.1)

            # Quality Metrics
            # FIXME: put into Analysis tools UI ?
            clusts = res['particles'][i]
            true_group_ids = get_cluster_label(data_blob['cluster_label'][i], clusts, column=6)
            pred_group_ids = res['inter_group_pred'][i]
            ari, ami, sbd, pur, eff = clustering_metrics(clusts, true_group_ids, pred_group_ids)

            # Count true stopping muons in the event
            for tp in true_particles:
                if tp.semantic_type != track_label: continue
                num_voxels = tp.size

                p = data_blob['particles_asis'][i][tp.id]
                if p.pdg_code() not in [13, -13]: continue
                endpoint = p.end_position()
                is_stopping = endpoint.x() >= 0 and endpoint.x() < spatial_size \
                            and endpoint.y() >= 0 and endpoint.y() < spatial_size \
                            and endpoint.z() >= 0 and endpoint.z() < spatial_size
                if not is_stopping: continue
                # Determine whether attached to Michel or not
                attached_to_Michel = False
                Michel_size = -1
                distance_to_Michel = -1
                for daughter_p in true_particles:
                    if daughter_p.semantic_type != 2: continue
                    daughter = data_blob['particles_asis'][i][daughter_p.id]
                    # #if daughter.id() == p.id() or daughter.parent_id() != p.id(): continue
                    # if daughter.pdg_code() not in [11, -11]: continue
                    # #if 'Decay' not in daughter.creation_process(): continue
                    # d = np.sqrt((daughter.position().x() - endpoint.x())**2 + (daughter.position().y() - endpoint.y())**2 + (daughter.position().z() - endpoint.z())**2)
                    # print('   BIP   ', daughter.creation_process(), daughter.track_id(), daughter.parent_track_id(), daughter.ancestor_track_id(), p.track_id(), p.ancestor_track_id(), daughter.parent_id(), p.id())
                    # print('found electron decay ', d, daughter.creation_process())
                    # print(daughter.position().x(), daughter.first_step().x())
                    # print(daughter.position().y(), daughter.first_step().y())
                    # print(daughter.position().z(), daughter.first_step().z())
                    # if d >= Michel_threshold: continue
                    if p.id() != daughter.parent_id(): continue
                    # print('found Michel in true particles !!!')
                    attached_to_Michel = True
                    Michel_size = daughter_p.size
                    distance_to_Michel = cdist(tp.points, daughter_p.points).min()
                    break
                # Determine whether it was matched
                is_matched = False
                for mp in matched_particles: # matching is done true to pred
                    if mp[0] is None or mp[0].id != tp.id: continue
                    is_matched = True
                    break

                true_muons.append(OrderedDict({
                    'index': index,
                    'attached_to_Michel': attached_to_Michel,
                    'distance_to_michel': distance_to_Michel,
                    'pdg': p.pdg_code(),
                    'num_voxels': num_voxels,
                    'Michel_num_voxels': Michel_size,
                    'is_matched': is_matched,
                    'overall_purity': pur,
                    'overall_efficiency': eff,
                    'overall_ari': ari
                }))

        # Loop over predicted particles
        for p in pred_particles:
            if p.semantic_type != track_label: continue
            coords = p.points

            # Check for presence of Michel electron
            attached_to_Michel = False
            closest_point = None
            for p2 in pred_particles:
                if p2.semantic_type != 2: continue
                d =  cdist(p.points, p2.points)
                if d.min() >= Michel_threshold: continue
                attached_to_Michel = True
                closest_point = d.min(axis=1).argmin()

            if not attached_to_Michel: continue

            # If asked to check predicted PID, exclude non-predicted-muons
            if pid_constraint and p.pid != muon_label: continue

            # PCA to get a rough direction
            coords_pca = pca.fit_transform(p.points)[:, 0]
            # Make sure where the end vs start is
            # if end == 0 we have the right bin ordering, otherwise might need to flip when we record the residual range
            distances_endpoints = [((coords[coords_pca.argmin(), :] - coords[closest_point, :])**2).sum(), ((coords[coords_pca.argmax(), :] - coords[closest_point, :])**2).sum()]
            end = np.argmin(distances_endpoints)

            # Record the stopping muon
            update_dict = {
                'index': index,
                'pred_particle_type': p.pid,
                'pred_particle_is_primary': p.is_primary,
                'pred_particle_size': p.size,
                #'projected_x_length': projected_x_length,
                'theta_yz': np.arctan2((coords[:, y].max() - coords[:, y].min()),(coords[:, z].max()-coords[:, z].min())),
                'theta_xz': np.arctan2((coords[:, x].max() - coords[:, x].min()),(coords[:, z].max()-coords[:, z].min())),
                'matched': False,
                'pca_length': coords_pca.max() - coords_pca.min(),
                #'t0': t0,
                'true_pdg': -1,
                'true_size': -1,
                'cluster_purity': -1,
                'cluster_efficiency': -1,
                'distance_endpoint_to_michel': np.sqrt(np.min(distances_endpoints))
                }
            if not data:
                for mp in matched_particles: # matching is done true2pred
                    if mp[1] is None or mp[1].id != p.id: continue
                    if mp[0] is None: continue
                    m = mp[0]
                    pe = m.purity_efficiency(p)
                    update_dict.update({
                        'matched': True,
                        'true_pdg': m.pid,
                        'true_size': m.size,
                        'cluster_purity': pe['purity'],
                        'cluster_efficiency': pe['efficiency']
                    })

            muons.append(OrderedDict(update_dict))
            track_dict= update_dict

            # Split into segments and compute local dQ/dx
            bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
            bin_inds = np.digitize(coords_pca, bins)

            # spatial_bins = np.arange(0, spatial_size, spatial_bin_size)
            # y_inds = np.digitize(coords[:, y], bins)
            # z_inds = np.digitize(coords[:, z], bins)
            # x_inds = np.digitize(coords[:, x], bins)
            for i in np.unique(bin_inds):
                mask = bin_inds == i
                if np.count_nonzero(mask) < 2: continue
                # Repeat PCA locally for better measurement of dx
                pca_axis = pca.fit_transform(p.points[mask])
                dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
                update_dict = OrderedDict({
                    'index': index,
                    'cell_dQ': p.depositions[mask].sum(),
                    'cell_dN':  np.count_nonzero(mask),
                    'cell_dx': dx,
                    'cell_bin': i,
                    'cell_residual_range': (i if end == 0 else len(bins)-i-1) * bin_size,
                    'nbins': len(bins)
                })
                update_dict.update(track_dict)
                muon_cells.append(update_dict)

    return [muon_cells, muons, true_muons]
