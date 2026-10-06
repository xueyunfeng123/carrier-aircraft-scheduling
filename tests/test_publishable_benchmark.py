"""Tests for canonical benchmark runtime controls."""

from __future__ import annotations

import os
import unittest

import torch

from scripts.run_publishable_benchmark import (
    EVALUATION_INTRAOP_THREADS,
    THREAD_ENVIRONMENT,
    configure_evaluation_threads,
)


class PublishableBenchmarkRuntimeTest(unittest.TestCase):
    def test_evaluation_runtime_is_single_threaded(self) -> None:
        configure_evaluation_threads()

        self.assertEqual(EVALUATION_INTRAOP_THREADS, 1)
        self.assertEqual(torch.get_num_threads(), 1)
        for name, value in THREAD_ENVIRONMENT.items():
            self.assertEqual(os.environ[name], value)


if __name__ == "__main__":
    unittest.main()
