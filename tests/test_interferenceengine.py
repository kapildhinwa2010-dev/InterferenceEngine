import pytest
import torch
from interferenceengine import InterferenceEngine, create_interferenceengine

class TestInterferenceEngine:
    def test_init(self):
        m = InterferenceEngine(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = InterferenceEngine(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_interferenceengine()
        assert isinstance(m, InterferenceEngine)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
