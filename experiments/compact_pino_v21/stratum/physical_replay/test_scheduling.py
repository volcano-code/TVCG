import json
from types import SimpleNamespace

import numpy as np

from . import assist_columns


def test_distributes_every_original_column_at_both_steps_and_keeps_a_source_refusal_empty(tmp_path):
    probes={}
    (tmp_path/"centers").mkdir()
    for name,dim,available in (("plane",3,True),("curved",8,True),("refused",8,False)):
        probes[name]=SimpleNamespace(chart=SimpleNamespace(basis=np.empty((10,dim))))
        (tmp_path/"centers"/(name+".json")).write_text(json.dumps(
            {"center_seconds":5.,"source":{"defined":True} if available else None,"primal":None}))
    jobs=assist_columns.schedule_pairs(probes,tmp_path)
    expected={f"{row}/{step}/{col}" for row,dim in (("plane",3),("curved",8))
              for step in (0,1) for col in range(dim)}
    assert len(jobs)==len(expected)
    assert set(jobs)==expected
