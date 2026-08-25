import importlib.util
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, ModelConfig, TrainingConfig, TinyGPT
        global BPETokenizer, LanguageModelTrainer, load_checkpoint
        import torch

        from deep_learning_studio.config import ModelConfig, TrainingConfig
        from deep_learning_studio.model import TinyGPT
        from deep_learning_studio.tokenizer import BPETokenizer
        from deep_learning_studio.training import LanguageModelTrainer, load_checkpoint

    def test_train_save_load_and_generate(self):
        corpus = "くらげちゃんは海を泳ぐ。今日は青い海です。\n" * 40
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, target_vocab_size=32, min_pair_frequency=1000)
        model = TinyGPT(
            ModelConfig(
                tokenizer.vocab_size,
                context_length=8,
                embedding_dim=16,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
            )
        )
        trainer = LanguageModelTrainer(
            model,
            tokenizer,
            corpus,
            TrainingConfig(
                max_steps=2,
                batch_size=2,
                warmup_steps=1,
                eval_interval=1,
                eval_batches=1,
                device="cpu",
            ),
        )

        history = trainer.train()

        self.assertEqual(len(history), 2)
        self.assertTrue(history[-1].attention)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            trainer.save_checkpoint(checkpoint_path)
            restored, restored_tokenizer, checkpoint = load_checkpoint(
                checkpoint_path, device="cpu"
            )
            prompt = torch.tensor([[restored_tokenizer.bos_id]], dtype=torch.long)
            output = restored.generate(prompt, 2, top_k=5)
            self.assertEqual(tuple(output.shape), (1, 3))
            self.assertEqual(checkpoint["step"], 2)


if __name__ == "__main__":
    unittest.main()
