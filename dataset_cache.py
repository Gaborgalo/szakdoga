"""Versioned model-input caches for the thesis notebook; no model weights or credentials.

The on-disk payload contains only primitive containers and CPU tensors. It is loaded
with torch's restricted weights_only unpickler; registered data classes are rebuilt
explicitly afterward. Tensor storage sharing (including overlapping windows) survives.
"""
from __future__ import annotations
import ast
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import re
import shutil
import zipfile
import numpy as np
import pandas as pd
import torch

FORMAT_VERSION = 2
NOTEBOOK_NAME = 'Gabor_Galovics_Thesis_work_3_FIXED(2).ipynb'
DATA_CLASSES = {'BuyTimeRecord', 'TabularBuyEpisode'}
# Stable source cell metadata IDs; populated by the notebook update script.
HELPER_IDS = ['0f1PugSYyE06', '810jqa5ur0HN', 'SGd0z2VVr0HP', 'K8RoQ9aGtbJ-', 'cde782f4', 'FV3nCwAar0HX', 'eRJ5kwO_r0Hc', 'HWpa7YkIr0He', 'd7dbdb5e0b7d']
class CacheMismatch(ValueError):
    pass

class ArrayVector:
    def __init__(self, array): self.array = array
    def toArray(self): return self.array


def _encode(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, np.generic): return _encode(value.item(), memo)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    identity = id(value)
    if identity in memo: return memo[identity]
    if isinstance(value, datetime): return {'_cache_type':'datetime','value':value.isoformat()}
    if isinstance(value, date): return {'_cache_type':'date','value':value.isoformat()}
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc': raise TypeError('Only numeric dataset arrays are supported.')
        result = {'_cache_type':'ndarray','value':torch.from_numpy(np.ascontiguousarray(value))}
    elif isinstance(value, (list, tuple)):
        result = [] if isinstance(value,list) else {'_cache_type':'tuple','value':[]}
        memo[identity] = result
        target = result if isinstance(result,list) else result['value']
        target.extend(_encode(item,memo) for item in value)
    elif isinstance(value, dict):
        result = {}
        memo[identity] = result
        for key,item in value.items():
            if not isinstance(key,(str,int)): raise TypeError('Cache dictionary keys must be strings or integers.')
            result[key] = _encode(item,memo)
    elif is_dataclass(value) and type(value).__name__ in DATA_CLASSES:
        result = {'_cache_type':type(value).__name__,'fields':{}}
        memo[identity] = result
        result['fields'].update({f.name:_encode(getattr(value,f.name),memo) for f in fields(value)})
    else:
        raise TypeError(f'Not model-input data: {type(value).__name__}. Models/loaders/secrets are never cached.')
    memo[identity] = result
    return result


def _decode(value, registry, memo=None):
    memo = {} if memo is None else memo
    if value is None or isinstance(value,(str,bool,int,float,torch.Tensor)): return value
    identity = id(value)
    if identity in memo: return memo[identity]
    if isinstance(value,list):
        result=[];memo[identity]=result
        result.extend(_decode(item,registry,memo) for item in value)
        return result
    kind=value.get('_cache_type') if isinstance(value,dict) else None
    if kind=='date': result=date.fromisoformat(value['value'])
    elif kind=='datetime': result=datetime.fromisoformat(value['value'])
    elif kind=='ndarray': result=value['value'].numpy()
    elif kind=='tuple': result=tuple(_decode(item,registry,memo) for item in value['value'])
    elif kind in DATA_CLASSES:
        cls=registry[kind]
        if set(value['fields'])!={f.name for f in fields(cls)}: raise CacheMismatch('Data-class schema changed.')
        result=cls(**{key:_decode(item,registry,memo) for key,item in value['fields'].items()})
    elif kind is not None: raise CacheMismatch(f'Unknown cache data type: {kind}')
    elif isinstance(value,dict):
        result={};memo[identity]=result
        result.update({key:_decode(item,registry,memo) for key,item in value.items()})
    else: raise CacheMismatch('Unexpected cache value.')
    memo[identity]=result
    return result


def save_cache(path, state, *, overwrite=False):
    """Save every prepared split in one Drive file, replacing it only after a full write."""
    path=Path(path)
    if path.is_dir():
        raise IsADirectoryError('An older cache directory exists here. Load it directly, or choose a different filename for a rebuild; existing data was preserved.')
    if path.exists() and not overwrite:
        raise FileExistsError('Cache exists; explicitly rebuild/overwrite it.')
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(format_version=FORMAT_VERSION,state=_encode(state))
    descriptor,temporary=tempfile.mkstemp(prefix='.prepared-',suffix='.pt',dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload,temporary)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(bytes=path.stat().st_size)


def load_cache(path, registry):
    """Load a single model-input file using Torch's restricted decoder."""
    if Path(path).is_dir():
        return _load_legacy_cache(path,registry)
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    if payload.get('format_version')!=FORMAT_VERSION:
        raise CacheMismatch('Unsupported dataset cache version; rebuild datasets.')
    return _decode(payload['state'],registry)


def _sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def read_manifest(path):
    manifest=json.loads((Path(path)/'manifest.json').read_text())
    if manifest.get('format_version')!=1: raise CacheMismatch('Unsupported dataset cache version.')
    if not manifest.get('parts'): raise CacheMismatch('Empty dataset archive.')
    names=[p['name'] for p in manifest['parts']]
    if len(names)!=len(set(names)) or any(not re.fullmatch(r'datasets-\d{5}\.part',name) for name in names):
        raise CacheMismatch('Invalid archive part names.')
    return manifest


def verify_cache(path, manifest=None):
    path=Path(path);manifest=manifest or read_manifest(path)
    for part in manifest['parts']:
        file=path/part['name']
        if not file.is_file() or file.is_symlink() or file.stat().st_size!=part['bytes'] or _sha(file)!=part['sha256']:
            raise CacheMismatch(f'Corrupt or changed cache part: {part["name"]}')
    return manifest


def _load_legacy_cache(path, registry):
    path=Path(path);manifest=read_manifest(path)
    verify_cache(path,manifest)
    with tempfile.TemporaryDirectory(prefix='dataset-cache-load-') as work:
        archive=Path(work)/'datasets.zip'
        with archive.open('wb') as out:
            for part in manifest['parts']:
                with (path/part['name']).open('rb') as inp: shutil.copyfileobj(inp,out)
        if _sha(archive)!=manifest['archive_sha256']: raise CacheMismatch('Archive checksum mismatch.')
        payload=Path(work)/'datasets.pt'
        with zipfile.ZipFile(archive) as z:
            if z.namelist()!=['datasets.pt'] or z.getinfo('datasets.pt').file_size!=manifest['payload_bytes']:
                raise CacheMismatch('Unexpected dataset archive contents.')
            with z.open('datasets.pt') as inp,payload.open('wb') as out: shutil.copyfileobj(inp,out)
        if _sha(payload)!=manifest['payload_sha256']: raise CacheMismatch('Payload checksum mismatch.')
        encoded=torch.load(payload,map_location='cpu',weights_only=True)
    return _decode(encoded,registry)


def source_cells(notebook):
    data=json.loads(Path(notebook).read_text())
    return {c.get('metadata',{}).get('id'):''.join(c['source']) for c in data['cells'] if c['cell_type']=='code'}


def unguarded_source(source):
    """Remove only our cache guard; preserve the original source for one-time building."""
    if '# DATASET-CACHE-GUARD' not in source: return source
    tree=ast.parse(source)
    node=tree.body[0]
    if not isinstance(node,ast.If): raise ValueError('Invalid dataset cache guard.')
    return ast.unparse(ast.Module(body=node.body,type_ignores=[]))+'\n'


def bootstrap_definitions(namespace,notebook):
    """Load data classes and model helpers without importing Spark or running preparation/training."""
    import builtins,copy,math,collections
    import torch.nn as nn
    import torch.optim as optim
    import xgboost as xgb
    from torch.utils.data import DataLoader,Dataset
    from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
    from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
    namespace.update({k:v for k,v in locals().items() if k not in ('namespace','notebook')})
    namespace.update(np=np,pd=pd,torch=torch)
    cells=source_cells(notebook)
    for identifier in HELPER_IDS:
        source=unguarded_source(cells[identifier])
        # Entire buy-time and Q helper cells contain only definitions/imports/constants.
        if source.startswith(('# Buy-time helpers:', '# Appended XGBoost / fitted-Q helpers.', '# Fixed parameters and independent')):
            exec(compile(source,str(notebook),'exec'),namespace)
        else:
            tree=ast.parse(source)
            nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.Import,ast.ImportFrom))]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),str(notebook),'exec'),namespace)
    namespace.setdefault('STACK_KEYS',['legId','flightDate','searchDate'])
    return {name:namespace[name] for name in DATA_CLASSES}


def _pack_frame(frame,X):
    return dict(columns={name:frame[name].tolist() for name in frame if name!='features'},X=X)


def _unpack_frame(value):
    frame=pd.DataFrame(value['columns'])
    frame['features']=[ArrayVector(row) for row in value['X']]
    return frame


def snapshot_model_data(ns):
    """Whitelisted model inputs only: no fitted models, optimizers, Spark sessions or environment."""
    price={}
    for suffix in ('','_no_ts'):
        for split,frame_name in [('train','xgb_train_pd'),('val','xgb_val_pd'),('test','common_test_pd')]:
            X=ns[f'X_{split}_np{suffix}'];y=ns[f'y_{split}_np{suffix}']
            frame=ns[frame_name+suffix]
            price[split+suffix]=dict(frame=_pack_frame(frame,X),X=X,y=y,
                                    anchors=frame[ns['ANCHOR_COL']].to_numpy())
    dl={}
    for split in ('train','val','test'):
        dl[split]=dict(X=ns[f'X_{split}_list_dl'],y=ns[f'y_{split}_tensor_dl'],
                       anchors=ns[f'{split}_anchor_np'],lengths=ns[f'lengths_{split}'],keys=ns[f'{split}_sequence_keys'])
    # Build stacking's later-validation inputs once; never train a model here.
    sequences=ns['prepare_ordered_sequences'](ns['df_processed_dl_all'],ns['stack_target_keys'])
    X,y,anchors,lengths,keys=ns['spark_sequences_to_variable_numpy'](sequences)
    frame=ns['aligned_tabular_rows'](ns['val_df_processed'],keys)
    matrix=np.vstack(frame['features'].map(lambda vector:vector.toArray()).to_numpy()).astype(np.float32)
    stack=dict(X=X,y=y,anchors=anchors,lengths=lengths,keys=keys,frame=_pack_frame(frame,matrix))
    buytime={}
    for horizon,bundle in ns['buytime_bundles'].items():
        buytime[horizon]={key:value for key,value in bundle.items() if key not in ('datasets','loaders')}
        buytime[horizon]['records']={split:ds.records for split,ds in bundle['datasets'].items()}
    names=('KEY_COLUMNS','TARGET_COL','ANCHOR_COL','LEVEL_COL','DELTA_STD','feature_dim','batch_size',
           'train_date_boundary','val_date_boundary','stack_date_boundary','SEQUENCE_LENGTH','N_DIFF_LAGS',
           'MATCH_TRAINING_TARGETS','DL_TRAIN_WINDOW_STRIDE','DL_MAX_TRAIN_WINDOWS_PER_ITINERARY',
           'bt_extra_stats','buytime_bounds','BUYTIME_HISTORY_LENGTH','BUYTIME_BATCH_SIZE',
           'FUTURE_HORIZON_VALUES','TABULAR_Q_MAX_TRAIN_EPISODES')
    metadata={name:ns[name] for name in names if name in ns}
    return dict(price=price,dl=dl,stack=stack,buytime=buytime,tabular_q=ns['tabular_q_data'],metadata=metadata)


def restore_model_data(state,ns):
    ns.update(state['metadata'])
    for suffix in ('','_no_ts'):
        for split,frame_name in [('train','xgb_train_pd'),('val','xgb_val_pd'),('test','common_test_pd')]:
            value=state['price'][split+suffix]
            ns[f'X_{split}_np{suffix}'],ns[f'y_{split}_np{suffix}']=value['X'],value['y']
            ns[frame_name+suffix]=_unpack_frame(value['frame'])
            if split=='test':ns['anchor_test_np'+suffix]=value['anchors']
    for split,value in state['dl'].items():
        ns[f'X_{split}_list_dl'],ns[f'y_{split}_tensor_dl']=value['X'],value['y']
        ns[f'{split}_anchor_np'],ns[f'lengths_{split}'],ns[f'{split}_sequence_keys']=value['anchors'],value['lengths'],value['keys']
        ds=ns['CustomSequenceDataset'](value['X'],value['y'],value['lengths'])
        ns[f'{split}_dataset_dl']=ds
        ns[f'{split}_loader_dl']=ns['DataLoader'](ds,batch_size=ns['batch_size'],shuffle=(split=='train'),collate_fn=ns['sequence_collate_fn'])
    ns['buytime_bundles']={}
    for horizon,saved in state['buytime'].items():
        # Preserve saved scaling/statistics instead of silently fitting anything during load.
        bundle={k:v for k,v in saved.items() if k!='records'}
        bundle['datasets']={split:ns['BuyTimeDataset'](records) for split,records in saved['records'].items()}
        bundle['loaders']={split:ns['DataLoader'](ds,batch_size=ns['BUYTIME_BATCH_SIZE'],shuffle=(split=='train'),collate_fn=ns['buytime_collate'])
                           for split,ds in bundle['datasets'].items()}
        ns['buytime_bundles'][horizon]=bundle
    ns['datasets_by_horizon']={h:b['datasets'] for h,b in ns['buytime_bundles'].items()}
    ns['dataloaders_by_horizon']={h:b['loaders'] for h,b in ns['buytime_bundles'].items()}
    ns['tabular_q_data']=state['tabular_q']
    ns['DATASET_CACHE_STACK']=state['stack']
    ns['val_df_processed']=_unpack_frame(state['stack']['frame'])  # stacking uses only these late-validation rows
    ns['test_df_processed']=ns['common_test_pd']
    ns['STACK_KEYS']=ns['KEY_COLUMNS']
    ns['device']=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name in ('buytime_results','tabular_xgb_fitted','tabular_q_fitted','tabular_sequential_results'):ns[name]={}
    for name in ('xgboost_price_metrics','xgboost_no_ts_price_metrics','price_metrics_gru','price_metrics_cnn',
                 'price_metrics_lstm','transformer_metrics','stacked_metrics','persistence_metrics'):
        ns.pop(name,None)
    ns['DATASET_CACHE_LOADED']=True


def build_model_data(ns,notebook):
    """Execute only preparation cells in dependency order, never model training/EDA plots/Optuna."""
    cells=source_cells(notebook)
    # Execute only data preparation, without fitting any models.
    for identifier in ns['DATASET_CACHE_BUILD_IDS']:
        source=unguarded_source(cells[identifier])
        exec(compile(source,str(notebook),'exec'),ns)
    return snapshot_model_data(ns)
