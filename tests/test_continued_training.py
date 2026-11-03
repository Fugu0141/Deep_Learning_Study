import importlib.util
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ContinuedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, ModelConfig, TrainingConfig, TinyGPT
        global BPETokenizer, LanguageModelTrainer, load_checkpoint
        import torch

        from deep_learning_studio.config import ModelConfig, TrainingConfig
        from deep_learning_studio.model import TinyGPT
        from deep_learning_studio.tokenizer import BPETokenizer
        from deep_learning_studio.training import LanguageModelTrainer, load_checkpoint

    def config(self):
        return TrainingConfig(
            max_steps=2,
            batch_size=2,
            warmup_steps=1,
            eval_interval=1,
            eval_batches=1,
            device="cpu",
        )

    def test_continue_after_checkpoint_with_new_japanese_characters(self):
        old_corpus = "ねこは海にいる。きょうは晴れです。\n" * 30
        new_corpus = "犬は森の中で遊ぶ。\n" * 30
        tokenizer = BPETokenizer()
        tokenizer.train(old_corpus, target_vocab_size=32, min_pair_frequency=1000)
        model = TinyGPT(
            ModelConfig(tokenizer.vocab_size, context_length=8,
                        embedding_dim=16, num_heads=4, num_layers=1, dropout=0.0)
        )
        first = LanguageModelTrainer(model, tokenizer, old_corpus, self.config())
        first.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            first.save_checkpoint(path)
            restored_model, restored_tokenizer, checkpoint = load_checkpoint(
                path, device="cpu"
            )
            self.assertEqual(checkpoint["step"], 2)
            old_ids = restored_tokenizer.encode("ねこは海にいる。")
            old_weight = restored_model.token_embedding.weight.detach().clone()
            old_size = restored_tokenizer.vocab_size

            additions = restored_tokenizer.extend_characters(new_corpus)
            self.assertIn("犬", additions)
            self.assertGreater(restored_tokenizer.vocab_size, old_size)
            self.assertEqual(restored_tokenizer.encode("ねこは海にいる。"), old_ids)
            self.assertEqual(restored_tokenizer.token_to_id["犬"], old_size + additions.index("犬"))

            restored_model.expand_vocabulary(restored_tokenizer.vocab_size)
            self.assertTrue(torch.equal(
                restored_model.token_embedding.weight[:old_size], old_weight
            ))
            self.assertIs(
                restored_model.token_embedding.weight,
                restored_model.language_head.weight,
            )
            continuation = LanguageModelTrainer(
                restored_model, restored_tokenizer, new_corpus, self.config(),
                probe_text="犬は森にいる。",
            )
            self.assertEqual(continuation.probe_coverage, 1.0)
            loss, ppl, coverage = continuation.evaluate_probe()
            self.assertIsNotNone(loss)
            self.assertGreater(ppl, 0)
            self.assertEqual(coverage, 1.0)
            continuation.train()
            self.assertEqual(continuation.current_step, 2)
            self.assertIsNotNone(continuation.history[-1].probe_perplexity)
            continuation.save_checkpoint(path)
            final_model, final_tokenizer, _ = load_checkpoint(path, device="cpu")
            self.assertEqual(final_model.config.vocab_size, final_tokenizer.vocab_size)

    def test_unknown_test_characters_are_not_scored_as_valid_japanese(self):
        corpus = "ねこは海にいる。\n" * 30
        tokenizer = BPETokenizer()
        tokenizer.train(corpus, target_vocab_size=32, min_pair_frequency=1000)
        model = TinyGPT(
            ModelConfig(tokenizer.vocab_size, context_length=8,
                        embedding_dim=16, num_heads=4, num_layers=1, dropout=0.0)
        )
        trainer = LanguageModelTrainer(
            model, tokenizer, corpus, self.config(), probe_text="象は空を飛ぶ。"
        )
        loss, ppl, coverage = trainer.evaluate_probe()
        self.assertIsNone(loss)
        self.assertIsNone(ppl)
        self.assertLess(coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
