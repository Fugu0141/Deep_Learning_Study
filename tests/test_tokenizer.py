import unittest

from deep_learning_studio.tokenizer import BPETokenizer


class BPETokenizerTests(unittest.TestCase):
    def test_round_trip_preserves_unicode_and_newlines(self):
        text = "くらげちゃんは海を泳ぐ。\nくらげちゃんは光る。\n"
        tokenizer = BPETokenizer()
        history = tokenizer.train(text, target_vocab_size=64, min_pair_frequency=2)

        encoded = tokenizer.encode(text, add_bos=True, add_eos=True)

        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertGreater(len(history), 0)
        self.assertEqual(encoded[0], tokenizer.bos_id)
        self.assertEqual(encoded[-1], tokenizer.eos_id)

    def test_unknown_character_uses_unknown_token(self):
        tokenizer = BPETokenizer()
        tokenizer.train("abcabc", target_vocab_size=12)

        encoded = tokenizer.encode("a雪")

        self.assertIn(tokenizer.unk_id, encoded)

    def test_serialization_keeps_encoding_stable(self):
        text = "deep learning deep learning"
        tokenizer = BPETokenizer()
        tokenizer.train(text, target_vocab_size=24)
        restored = BPETokenizer.from_dict(tokenizer.to_dict())

        self.assertEqual(restored.encode(text), tokenizer.encode(text))
        self.assertEqual(restored.decode(restored.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
