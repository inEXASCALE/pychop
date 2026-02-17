# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import pandas
from typing import Callable
from IPython.display import HTML


def pandas_render(df: pandas.DataFrame, **kwargs) -> HTML:
    """
    Render a dataframe, hiding the index,
    and set ID to minimize diffs for notebook regression tests
    """
    s = df.style.hide().set_uuid("my_id")
    for f, v in kwargs.items():
        if isinstance(getattr(s, f, None), Callable):
            s = getattr(s, f)(v)
        else:
            s = s.format(**{f: v})
    return HTML(s.to_html())
