import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TinyGPTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, ModelConfig, TinyGPT
        import torch

        from deep_learning_studio.config import ModelConfig
        from deep_learning_studio.model import TinyGPT

    def test_forward_loss_and_attention_shapes(self):
        config = ModelConfig(
            vocab_size=20,
            context_length=8,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
        model = TinyGPT(config)
        inputs = torch.randint(0, config.vocab_size, (2, config.context_length))
        output = model(inputs, inputs, capture_attention=True)

        self.assertEqual(tuple(output.logits.shape), (2, 8, 20))
        self.assertIsNotNone(output.loss)
        self.assertEqual(len(output.attentions), 2)
        self.assertEqual(tuple(output.attentions[0].shape), (2, 4, 8, 8))

    def test_attention_cannot_look_ahead(self):
        config = ModelConfig(20, 8, 16, 4, 1, 0.0)
        model = TinyGPT(config)
        inputs = torch.randint(0, 20, (1, 8))
        attention = model(inputs, capture_attention=True).attentions[0]
        future_mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)

        self.assertTrue(torch.all(attention[0, 0][future_mask] == 0))


if __name__ == "__main__":
    unittest.main()
