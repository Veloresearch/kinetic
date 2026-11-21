"""Smoke tests for the installed wheel.

KINETIC_TEST_GGUF=SmolLM2-135M-Instruct-f16.gguf KINETIC_TEST_EMBED_GGUF=bge-small-en-v1.5-f16.gguf
python -m unittest discover -s tests
"""

import math
import os
import struct
import threading
import unittest

import kinetic

TEXT = os.environ.get("KINETIC_TEST_GGUF")
EMBED = os.environ.get("KINETIC_TEST_EMBED_GGUF")


class Errors(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(RuntimeError) as e:
            kinetic.Model("no-such-model.gguf")
        self.assertIn("no-such-model.gguf", str(e.exception))

    def test_version(self):
        self.assertRegex(kinetic.__version__, r"^\d+\.\d+\.\d+")


@unittest.skipUnless(TEXT, "set KINETIC_TEST_GGUF")
class TextModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kinetic.Model(TEXT)

    def test_greedy_chat_is_repeatable_and_streams_the_same_text(self):
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        answer = self.model.chat(messages, max_tokens=16, temperature=0)
        self.assertTrue(answer.strip())
        self.assertEqual(answer, self.model.chat(messages, max_tokens=16, temperature=0))
        streamed = "".join(self.model.chat(messages, max_tokens=16, temperature=0, stream=True))
        self.assertEqual(streamed, answer)

    def test_seeded_generation_repeats(self):
        a = self.model.generate("Once upon a time", max_tokens=24, temperature=0.9, seed=7)
        b = self.model.generate("Once upon a time", max_tokens=24, temperature=0.9, seed=7)
        self.assertEqual(a, b)

    def test_a_dropped_stream_frees_the_model(self):
        stream = self.model.generate("Count: 1, 2, 3,", max_tokens=200, stream=True)
        next(stream)
        del stream
        self.assertTrue(self.model.generate("Hello", max_tokens=4))

    def test_threads_take_turns(self):
        out = []
        threads = [
            threading.Thread(target=lambda: out.append(self.model.generate("Hi", max_tokens=8)))
            for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(out), 3)

    def test_bad_messages_and_embed_fail(self):
        with self.assertRaises(ValueError):
            self.model.chat([{"role": "user"}])
        with self.assertRaises(RuntimeError):
            self.model.embed(["text"])


@unittest.skipUnless(TEXT, "set KINETIC_TEST_GGUF")
class Logits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = kinetic.Model(TEXT)

    def test_logits_shape_and_stability(self):
        row = self.model.logits("Once upon a time")
        self.assertEqual(len(row), self.model.n_vocab)
        self.assertTrue(all(map(math.isfinite, row)))
        self.assertEqual(row, self.model.logits("Once upon a time"))

    def test_logits_rows_are_flat_little_endian_bytes(self):
        ids = self.model.tokenize("The capital of France is", special=True)
        b = self.model.logits_rows(ids)
        n = self.model.n_vocab
        self.assertEqual(len(b), 4 * len(ids) * n)
        # the row after the prompt, which logits returns as floats, is the
        # last row of the buffer
        row = self.model.logits("The capital of France is")
        last = struct.unpack_from("<f", b, 4 * (len(ids) - 1) * n)[0]
        self.assertEqual(last, row[0])
        self.assertTrue(
            all(
                math.isfinite(struct.unpack_from("<f", b, i)[0])
                for i in (0, 4, 4 * n, len(b) - 4)
            )
        )

    def test_logits_rows_reject_bad_input(self):
        with self.assertRaises(RuntimeError):
            self.model.logits_rows([])
        with self.assertRaises(RuntimeError) as e:
            self.model.logits_rows([2**31 - 1])
        self.assertIn("past the vocabulary", str(e.exception))


@unittest.skipUnless(EMBED, "set KINETIC_TEST_EMBED_GGUF")
class EmbeddingLogits(unittest.TestCase):
    def test_no_logits_on_an_embedding_model(self):
        model = kinetic.Model(EMBED)
        self.assertIsNone(model.n_vocab)
        with self.assertRaises(RuntimeError) as e:
            model.logits("a cat")
        self.assertIn("embedding", str(e.exception))


@unittest.skipUnless(EMBED, "set KINETIC_TEST_EMBED_GGUF")
class EmbeddingModel(unittest.TestCase):
    def test_vectors(self):
        model = kinetic.Model(EMBED)
        a, b = model.embed(["a cat", "a dog"])
        self.assertEqual(len(a), len(b))
        self.assertGreater(len(a), 0)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
