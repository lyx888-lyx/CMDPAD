"""Small CPU smoke test for missing helper and checkpoint restoration."""
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model import MultimodalAttentionModel
from utils import build_model_and_load, infer_dims_from_feature, set_seed


class CheckRestoration(unittest.TestCase):
    def test_restore_personality_and_modality_order(self):
        set_seed(42)
        dims = {'bert-base-chinese': 4, 'convnext-base': 3}
        model = MultimodalAttentionModel(
            dims, proj_dim=8, transformer_layers=1, transformer_heads=2,
            dropout=0, head_hidden=(8, 4), output_dim=5).eval()
        x = {m: torch.randn(2, d) for m, d in dims.items()}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'personality.pt'
            torch.save(model.state_dict(), path)
            args = types.SimpleNamespace(proj_dim=8, transformer_layers=1,
                                         transformer_heads=2, dropout=0,
                                         hidden=[8, 4])
            loaded = build_model_and_load(path, dims, args)
            self.assertEqual(tuple(loaded(x).shape), (2, 5))
            torch.testing.assert_close(model(x), loaded(x))
            with self.assertRaises(ValueError):
                build_model_and_load(path, dict(reversed(list(dims.items()))), args)

    def test_paired_dimension_inference(self):
        df = pd.DataFrame({
            'A_bert-base-chinese': [np.ones(4)],
            'B_bert-base-chinese': [np.ones(6)],
        })
        self.assertEqual(infer_dims_from_feature(df, ['bert-base-chinese'], 'A_'),
                         {'bert-base-chinese': 4})
        self.assertEqual(infer_dims_from_feature(df, ['bert-base-chinese'], 'B_'),
                         {'bert-base-chinese': 6})


if __name__ == '__main__':
    unittest.main()
