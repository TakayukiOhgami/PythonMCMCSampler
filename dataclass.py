import numpy as np
from copy import deepcopy


def _hstack(data1, data2):
    if data1 is None:
        return data2
    elif data2 is None:
        return data1
    else:
        return np.hstack((data1, data2))


class DataClass:
    def __init__(self, name='name', x=[], y=[], z=None,
                 xerr=None, yerr=None, zerr=None, tag=None):
        self.data = {name: self.EachData(x, y, z, xerr, yerr, zerr, tag)}

    def name(self):
        return list(self.data.keys())

    class EachData:
        def __init__(self, x, y, z, xerr, yerr, zerr, tag):
            self.x = np.array(x)
            self.y = np.array(y)
            self.z = None if z is None else np.array(z)
            self.xerr = None if xerr is None else np.array(xerr)
            self.yerr = None if yerr is None else np.array(yerr)
            self.zerr = None if zerr is None else np.array(zerr)
            self.tag = None if tag is None else np.array(tag, dtype=object)

    def xsort(self, inverse=False):
        for _name in self.name():
            _sd = self.data[_name]
            if _sd.tag is None:
                _sorted_idx = \
                    np.argsort(-_sd.x) if inverse else np.argsort(_sd.x)
                _sd.x = _sd.x[_sorted_idx]
                _sd.y = _sd.y[_sorted_idx]
                _sd.z = None if _sd.z is None else _sd.z[_sorted_idx]
                _sd.xerr = None if _sd.xerr is None else _sd.xerr[_sorted_idx]
                _sd.yerr = None if _sd.yerr is None else _sd.yerr[_sorted_idx]
                _sd.zerr = None if _sd.zerr is None else _sd.zerr[_sorted_idx]
            else:
                _temp = dict()
                for _idx, _tag in enumerate(np.unique(_sd.tag)):
                    _temp[_tag] = dict()
                    for _k, _a in _sd.__dict__.items():
                        _temp[_tag][_k] = None if _a is None else \
                            np.array([_v for _t, _v in zip(_sd.tag, _a)
                                      if _t == _tag])
                    _sorted_idx = np.argsort(-_temp[_tag]['x']) if inverse \
                        else np.argsort(_temp[_tag]['x'])
                    for _k in _temp[_tag].keys():
                        _temp[_tag][_k] = None if _temp[_tag][_k] is None \
                            else _temp[_tag][_k][_sorted_idx]
                    if _idx == 0:
                        _ret = {_k: _a for _k, _a in _temp[_tag].items()}
                    else:
                        for _k, _a in _temp[_tag].items():
                            _ret[_k] = _hstack(_ret[_k], _a)
                _sd.x = _ret['x']
                _sd.y = _ret['y']
                _sd.z = _ret['z']
                _sd.xerr = _ret['xerr']
                _sd.yerr = _ret['yerr']
                _sd.zerr = _ret['zerr']
                _sd.tag = _ret['tag']
        return self

    def combine(self, other):
        _sd, _od = self.data, other.data
        _temp_self = self.name()
        _temp_other = other.name()
        for _name in set(_temp_self+_temp_other):
            if _name in _temp_self and _name in _temp_other:
                _sd[_name].x = _hstack(_sd[_name].x, _od[_name].x)
                _sd[_name].y = _hstack(_sd[_name].y, _od[_name].y)
                _sd[_name].z = _hstack(_sd[_name].z, _od[_name].z)
                _sd[_name].xerr = _hstack(_sd[_name].xerr, _od[_name].xerr)
                _sd[_name].yerr = _hstack(_sd[_name].yerr, _od[_name].yerr)
                _sd[_name].zerr = _hstack(_sd[_name].zerr, _od[_name].zerr)
                _sd[_name].tag = _hstack(_sd[_name].tag, _od[_name].tag)
            else:
                _temp = _sd[_name] if _name in _temp_self else _od[_name]
                self.data[_name] = self.EachData(_temp.x, _temp.y, _temp.z,
                                                 _temp.xerr, _temp.yerr,
                                                 _temp.zerr, _temp.tag)
            self.xsort()
        return self


def copy(data):
    for _idx, _name in enumerate(data.name()):
        _temp = DataClass(name=deepcopy(_name),
                          x=np.copy(data.data[_name].x),
                          y=np.copy(data.data[_name].y),
                          z=np.copy(data.data[_name].z),
                          xerr=np.copy(data.data[_name].xerr),
                          yerr=np.copy(data.data[_name].yerr),
                          zerr=np.copy(data.data[_name].zerr),
                          tag=np.copy(data.data[_name].tag))
        if _idx == 0:
            _ret = _temp
        else:
            _ret = _ret.combine(_temp)
    return _ret


def pick(data, name=None, tag=None):
    _ret = copy(data) if name is None \
        else DataClass(name=name,
                       x=data.data[name].x,
                       y=data.data[name].y,
                       z=data.data[name].z,
                       xerr=data.data[name].xerr,
                       yerr=data.data[name].yerr,
                       zerr=data.data[name].zerr,
                       tag=data.data[name].tag)
    if tag is not None:
        for _name in _ret.name():
            _temp = dict()
            for _k, _a in _ret.data[_name].__dict__.items():
                _temp[_k] = \
                    np.array([_v for _t, _v in zip(data.data[_name].tag, _a)
                              if _t == tag]) \
                    if _a is not None else None
            _ret.data[_name].x = _temp['x']
            _ret.data[_name].y = _temp['y']
            _ret.data[_name].z = _temp['z']
            _ret.data[_name].xerr = _temp['xerr']
            _ret.data[_name].yerr = _temp['yerr']
            _ret.data[_name].zerr = _temp['zerr']
            _ret.data[_name].tag = _temp['tag']
    return _ret


if __name__ == '__main__':
    pass
