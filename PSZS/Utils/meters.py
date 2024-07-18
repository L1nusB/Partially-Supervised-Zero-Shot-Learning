from typing import Optional, Sequence, Any
import numpy as np
import warnings
import torch

class _BaseMeter:
    """Base class giving base syntax to all meters"""

    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt if fmt else ''
        self.exclude_reset = ('name', 'fmt', 'exclude_reset')
        self.reset()

    def reset(self) -> None:
        reset_keys = [k for k in self.__dict__.keys()
                      if k not in self.exclude_reset]
        for key in reset_keys:
            self.__dict__[key] = 0
    
    def reset_batch(self) -> None:
        return

    def update(self) -> None:
        raise NotImplementedError
    
    def get_avg(self) -> float:
        raise NotImplementedError

    def __str__(self) -> str:
        return f'_BaseMeter: {self.name}'

class StatsMeter(_BaseMeter):
    r"""
    Extension of tllib.utils.meter.AverageMeter that allows to specify which values to display.
    Available values are 'total', 'min', 'max', 'last', 'avg'
    """
    @staticmethod
    def get_average_meter(name: str, 
                          fmt: Optional[str] = ':f',
                          include_last: bool = False, 
                          include_total: bool = False,
                          include_batch: bool = False):
        includes = ['avg']
        if include_last:
            includes += ['last']
        if include_total:
            includes += ['total']
        if include_batch:
            includes += ['batch']
        return StatsMeter.get_stats_meter(name, fmt, includes=includes)

    @staticmethod
    def get_stats_meter_all(name: str, fmt: Optional[str] = ':f'):
        return StatsMeter.get_stats_meter(name, fmt, includes='all')
    
    @staticmethod
    def get_stats_meter_time(name: str, 
                             fmt: Optional[str] = ':f', 
                             reduce: bool = True,
                             show_batch: bool = True):
        if reduce:
            return StatsMeter.get_average_meter(name, fmt, 
                                                include_total=True, 
                                                include_batch=show_batch)
        else:
            if show_batch:
                excludes = 'last'
            else:
                excludes = ['last', 'batch']
                
            return StatsMeter.get_stats_meter(name, fmt, excludes=excludes)
        
    @staticmethod
    def get_stats_meter_min_max(name: str, fmt: Optional[str] = ':f'):
        return StatsMeter.get_stats_meter(name, fmt, 
                                          includes=['avg', 'min', 'max'])
        
    @staticmethod
    def get_stats_meter(name: str, fmt: Optional[str] = ':f', 
                        excludes: Optional[str|Sequence[str]] = None,
                        includes: Optional[str|Sequence[str]] = None):
        # Set Default values
        show_dict = {
                'total':False,
                'min':False,
                'max':False,
                'last':False,
                'avg':True,
                'batch':False
            }
        if excludes is not None and includes is not None:
            # Both excludes and includes are given. Includes take priority
            # If value in neither the default is used
            for exclude in excludes:
                show_dict[exclude] = False
            for include in includes:
                show_dict[include] = True
        elif excludes:
            # Only exludes given set default to all True
            show_dict = {
                'total':True,
                'min':True,
                'max':True,
                'last':True,
                'avg':True,
                'batch':True
            }
            if isinstance(excludes, str):
                if excludes in show_dict:
                    show_dict[excludes] = False
            else:
                for exclude in excludes:
                    if exclude in show_dict:
                        show_dict[exclude] = False
        elif includes:
            # Only includes given set default to all False
            show_dict = {
                'total':False,
                'min':False,
                'max':False,
                'last':False,
                'avg':False,
                'batch':False
            }
            if isinstance(includes, str):
                if includes in show_dict:
                    show_dict[includes] = True
                elif includes == 'all':
                   show_dict = {'total':True,
                                'min':True,
                                'max':True,
                                'last':True,
                                'avg':True,
                                'batch':True
                                } 
            else:
                for include in includes:
                    if include in show_dict:
                        show_dict[include] = True
        show_dict = {f'show_{k}':v for k,v in show_dict.items()}
        return StatsMeter(name, fmt, **show_dict)
    
    
    def __init__(self, 
                 name: str, 
                 fmt: Optional[str] = ':f', 
                 show_total: bool = False,
                 show_min: bool = False, 
                 show_max: bool = False, 
                 show_last: bool = False,
                 show_avg: bool = True,
                 show_batch: bool = False):
        super().__init__(name, fmt)
        self.watched_values = []
        self.watched_values = self.watched_values + ['last'] if show_last else self.watched_values
        self.watched_values = self.watched_values + ['batch'] if show_batch else self.watched_values
        self.watched_values = self.watched_values + ['total'] if show_total else self.watched_values
        self.watched_values = self.watched_values + ['avg'] if show_avg else self.watched_values
        self.watched_values = self.watched_values + ['min'] if show_min else self.watched_values
        self.watched_values = self.watched_values + ['max'] if show_max else self.watched_values
        self.last = 0
        self.batch = 0
        self.avg = 0
        self.total = 0
        self.count = 0
        self.min = np.inf
        self.max = -np.inf
        # Ignore min and max as they are set with other values than zero
        self.exclude_reset += ('watched_values','min', 'max')
        
    def reset(self) -> None:
        super().reset()
        self.min = np.inf
        self.max = -np.inf
        
    def reset_batch(self) -> None:
        self.batch = 0

    def update(self, val, n=1) -> None:
        # If n is 0 do nothing
        if n == 0:
            return
        self.last = val
        self.total += val * n
        self.batch += val * n
        self.count += n
        # One could scale value here to relative w.r.t 1 
        # but would probably lead only to more confusion
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        if self.count > 0:
            self.avg = self.total / self.count
            
    def get_avg(self) -> float:
        if 'avg' in self.watched_values:
            if isinstance(self.avg, float) or isinstance(self.avg, int):
                return self.avg
            elif isinstance(self.avg, torch.Tensor):
                return self.avg.item()
            elif isinstance(self.avg, np.ndarray):
                return self.avg.item()
        else:
            warnings.warn('avg is not a watched value for this meter.')
            return 0
        
    def get_last(self) -> float:
        if 'last' in self.watched_values:
            val = self.last
            if isinstance(val, float):
                return val
            elif isinstance(val, torch.Tensor):
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.item()
        else:
            warnings.warn('last is not a watched value for this meter.')
            return 0
            
    def __str__(self):
        displayVals = ['{{{key}{fmt}}}({key})'.format(key=key, fmt=self.fmt) for key in self.watched_values]
        fmtstr = '{name}: '+ " ".join(displayVals)
        return fmtstr.format(**self.__dict__)
    
    
class DynamicStatsMeter(_BaseMeter):
    @staticmethod
    def get_average_meter(name: str, 
                          fields: str|Sequence[str], 
                          fmt: Optional[str] = ':f', 
                          include_last: bool = False, 
                          include_total: bool = False,
                          include_batch: bool = False):
        includes = ['avg']
        if include_last:
            includes += ['last']
        if include_total:
            includes += ['total']
        if include_batch:
            includes += ['batch']
        return DynamicStatsMeter.get_stats_meter(name, fields, fmt, includes=includes)

    @staticmethod
    def get_stats_meter_all(name: str, fields: str|Sequence[str], fmt: Optional[str] = ':f'):
        return DynamicStatsMeter.get_stats_meter(name, fields, fmt, includes='all')
    
    @staticmethod
    def get_stats_meter_time(name: str, fields: str|Sequence[str], fmt: Optional[str] = ':f', 
                             reduce: bool = True, show_batch: bool = True):
        if reduce:
            return DynamicStatsMeter.get_average_meter(name, 
                                                       fields, 
                                                       fmt, 
                                                       include_total=True,
                                                       include_batch=show_batch)
        else:
            if show_batch:
                excludes = 'last'
            else:
                excludes = ['last', 'batch']
                
            return DynamicStatsMeter.get_stats_meter(name, fields, fmt, excludes=excludes)
        
    @staticmethod
    def get_stats_meter_min_max(name: str, fields: str|Sequence[str], fmt: Optional[str] = ':f'):
        return DynamicStatsMeter.get_stats_meter(name, fields, fmt, 
                                                 includes=['avg', 'min', 'max'])
        
    @staticmethod
    def get_stats_meter(name: str, fields: str|Sequence[str], fmt: Optional[str] = ':f', 
                        excludes: Optional[str|Sequence[str]] = None,
                        includes: Optional[str|Sequence[str]] = None):
        # Set Default values
        show_dict = {
                'total':False,
                'min':False,
                'max':False,
                'last':False,
                'avg':True,
                'batch':False,
            }
        if excludes is not None and includes is not None:
            # Both excludes and includes are given. Includes take priority
            # If value in neither the default is used
            for exclude in excludes:
                show_dict[exclude] = False
            for include in includes:
                show_dict[include] = True
        elif excludes:
            # Only exludes given set default to all True
            show_dict = {
                'total':True,
                'min':True,
                'max':True,
                'last':True,
                'avg':True,
                'batch':True,
            }
            if isinstance(excludes, str):
                if excludes in show_dict:
                    show_dict[excludes] = False
            else:
                for exclude in excludes:
                    if exclude in show_dict:
                        show_dict[exclude] = False
        elif includes:
            # Only includes given set default to all False
            show_dict = {
                'total':False,
                'min':False,
                'max':False,
                'last':False,
                'avg':False,
                'batch':False,
            }
            if isinstance(includes, str):
                if includes in show_dict:
                    show_dict[includes] = True
                elif includes == 'all':
                   show_dict = {'total':True,
                                'min':True,
                                'max':True,
                                'last':True,
                                'avg':True,
                                'batch':True,
                                } 
            else:
                for include in includes:
                    if include in show_dict:
                        show_dict[include] = True
        show_dict = {f'show_{k}':v for k,v in show_dict.items()}
        return DynamicStatsMeter(name, fields, fmt, **show_dict)
    
    def __init__(self, 
                 name: str, 
                 fields: str|Sequence[str], 
                 fmt: Optional[str] = ':f',  
                 show_total: bool = False,
                 show_min: bool = False, 
                 show_max: bool = False, 
                 show_last: bool = False,
                 show_avg: bool = True, 
                 show_batch: bool = False):
        super().__init__(name, fmt)
        self.watched_specifiers = []
        self.watched_specifiers = self.watched_specifiers + ['last'] if show_last else self.watched_specifiers
        self.watched_specifiers = self.watched_specifiers + ['batch'] if show_batch else self.watched_specifiers
        self.watched_specifiers = self.watched_specifiers + ['total'] if show_total else self.watched_specifiers
        self.watched_specifiers = self.watched_specifiers + ['avg'] if show_avg else self.watched_specifiers
        self.watched_specifiers = self.watched_specifiers + ['min'] if show_min else self.watched_specifiers
        self.watched_specifiers = self.watched_specifiers + ['max'] if show_max else self.watched_specifiers
        self.watched_fields = fields
        self.exclude_reset += ('watched_specifiers','watched_fields')
        
        for field in self.watched_fields:
            self.__dict__[f'last_{field}'] = 0
            self.__dict__[f'batch_{field}'] = 0
            self.__dict__[f'count_{field}'] = 0
            self.__dict__[f'total_{field}'] = 0
            self.__dict__[f'avg_{field}'] = 0
            self.__dict__[f'min_{field}'] = np.inf
            self.__dict__[f'max_{field}'] = -np.inf

    def reset(self) -> None:
        super().reset()
        # Reset min and max which have been set to 0 in super function call
        for field in getattr(self, 'watched_fields', []):
            self.__dict__[f'min_{field}'] = np.inf
            self.__dict__[f'max_{field}'] = -np.inf
            
    def reset_batch(self) -> None:
        for field in self.watched_fields:
            self.__dict__[f'batch_{field}'] = 0

    def update(self, vals:Any|Sequence[Any], n:int|Sequence[int]=1):
        # If n is 0 do nothing
        if n == 0:
            return
        assert len(vals)==len(self.watched_fields), \
            f"Number of given values {len(vals)} does not match watched fields {len(self.watched_fields)}"
        # If single number is given expand to all given values
        if isinstance(n, int):
            n = [n]*len(vals)
        else:
            assert len(n)==len(vals), f'Number of counts {len(n)} does not match number of values {len(vals)}'
            
        for index, field in enumerate(self.watched_fields):
            # If n is 0 do nothing
            if n[index] == 0:
                continue
            self.__dict__[f'last_{field}'] = vals[index]
            self.__dict__[f'count_{field}'] += n[index]
            self.__dict__[f'total_{field}'] += vals[index]*n[index]
            self.__dict__[f'batch_{field}'] += vals[index]*n[index]
            self.__dict__[f'min_{field}'] = min(self.__dict__[f'min_{field}'], vals[index])
            self.__dict__[f'max_{field}'] = max(self.__dict__[f'max_{field}'], vals[index])
            if vals[index] > 0:
                self.__dict__[f'avg_{field}'] = self.__dict__[f'total_{field}'] / self.__dict__[f'count_{field}']
            
    def get_avg(self, field: str) -> float:
        if 'avg' in self.watched_specifiers:
            val = self.__dict__[f'avg_{field}']
            if isinstance(val, float):
                return val
            elif isinstance(val, torch.Tensor):
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.item()
        else:
            warnings.warn('avg is not a watched specifier for this meter.')
            return 0
        
    def get_last(self, field: str) -> float:
        if 'last' in self.watched_specifiers:
            val = self.__dict__[f'last_{field}']
            if isinstance(val, float):
                return val
            elif isinstance(val, torch.Tensor):
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.item()
        else:
            warnings.warn('last is not a watched specifier for this meter.')
            return 0
    
    def __str__(self):
        # Creates list of prepared specifiers.
        # Curly braces are needed as they get populated by a later .format() call
        # Each field contains all relevant specifiers given in the form: ([] denotes variable while {} and () are included literally)
        # {[specifier1]_[field1]:[fmt]}([specifier1]) {[specifier2]_[field1]:[fmt]}([specifier2]) ...
        # Each such descriptors gets wrapped into square brackets and prefixed with the field name
        # [field1: ...]
        prep_specifiers = [f'[{field}: '+ 
                           " ".join(['{{{specifier}_[field]{fmt}}}({specifier})'.format(specifier=specifier, fmt=self.fmt) 
                                     for specifier in self.watched_specifiers]).replace('[field]', field) + 
                           "]" for field in self.watched_fields]
        
        fmtstr = '{name}: '+ " ".join(prep_specifiers)
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter:
    def __init__(self, num_batches: int, 
                 meters: Sequence[_BaseMeter], 
                 batch_meters: Optional[Sequence[_BaseMeter]]=None, 
                 exclude_simple_reset: Optional[Sequence[_BaseMeter]] = None,
                 prefix : str=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.batch_meters = [meter.name for meter in batch_meters] if batch_meters else []
        self.exclude_simple_reset = [meter.name for meter in exclude_simple_reset] if exclude_simple_reset else []
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  |  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    def add_meter(self, 
                  meters: _BaseMeter | Sequence[_BaseMeter], 
                  batch_meter: bool = False,
                  exclude_simple_reset: bool = False):
        if isinstance(meters, _BaseMeter):
            if meters in self.meters:
                print(f"Meter with name {meters.name} already in meters. Nothing added.")
            else:
                self.meters.append(meters)
            if batch_meter:
                self.batch_meters.append(meters.name)
            if exclude_simple_reset:
                self.exclude_simple_reset.append(meters.name)
        else:
            new_meters = [m for m in meters if m not in self.meters]
            self.meters += new_meters
            if batch_meter:
                self.batch_meters += [m.name for m in new_meters]
            if exclude_simple_reset:
                self.exclude_simple_reset += [m.name for m in new_meters]
                
    def remove_meter(self, 
                     meter: _BaseMeter | str):
        if isinstance(meter, _BaseMeter):
            meter_name = meter.name
        else:
            meter_name = meter
        if any(meter_name == m.name for m in self.meters)==False:
            print(f"Meter with name {meter_name} not found in meters. Nothing removed.")
            # No need to try and remove meter (does not exist)
            return
        self.meters = [m for m in self.meters if m.name != meter_name]
        if meter.name in self.batch_meters:
            self.batch_meters.remove(meter.name)
        if meter.name in self.exclude_simple_reset:
            self.exclude_simple_reset.remove(meter.name)
    
    def set_num_batches(self, num_batches: int):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)

    def reset(self, reset_all: bool = True):
        for meter in self.meters:
            if reset_all:
                meter.reset()
            else:
                if meter.name not in self.exclude_simple_reset:
                    if meter.name in self.batch_meters:
                        meter.reset_batch()
                    else:
                        meter.reset()