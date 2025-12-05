import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union

def sanitize_for_json(obj: Any) -> Any:
    """
    递归处理数据，确保能够被JSON序列化：
    1. 将NaN/Inf转换为None
    2. 将numpy类型转换为Python原生类型
    3. 将pandas类型转换为Python原生类型
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, (pd.Series, pd.Index)):
        return [sanitize_for_json(x) for x in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return [sanitize_for_json(x) for x in obj.to_dict(orient='records')]
    elif isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(x) for x in obj)
    return obj
