# Copyright (c) vLLM contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from: NGramProposer in https://github.com/vllm-project/vllm
# By happierpig on 04-08-2025

import numpy as np
from numba import jit


class LoopDetector:
    def __init__(self, window_width: int = 2048, n: int = 13, max_limits: int = 10):
        # Only detect within the recent n tokens
        self.window_width = window_width
        # n-gram size. Chosen as 13 following:
        # https://arxiv.org/pdf/2311.04850.pdf
        self.n = n
        # Maximal repeat count, larger to avoid false positives
        self.max_limits = max_limits

        assert self.window_width > self.n

    def detect(self, context_token_ids: np.ndarray) -> bool:
        """
        Detect if the context contains repeated n-grams, i.e. deadlock
        Args:
            context_token_ids (np.ndarray): The token ids of the context.
        Returns:
            bool: True if there are repeated n-grams, False otherwise.
        """
        return (
            _find_subarray_kmp_count(context_token_ids[-self.window_width :], self.n)
            > self.max_limits
        )


@jit(nopython=True)
def _kmp_lps_array(pattern: np.ndarray) -> np.ndarray:
    """
    Build the lps (longest proper prefix which is also suffix)
    array for the pattern.
    """
    lps = np.zeros(len(pattern), dtype=np.int32)
    prev_lps = 0  # length of the previous longest prefix suffix
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[prev_lps]:
            prev_lps += 1
            lps[i] = prev_lps
            i += 1
        else:
            if prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


@jit(nopython=True)
def _find_subarray_kmp_count(
    context_token_ids: np.ndarray,
    n: int,
) -> int:
    context_len = context_token_ids.shape[0]
    assert n > 0

    pattern = context_token_ids[-n:]
    lps = _kmp_lps_array(pattern)

    match_count = 0
    i = 0
    j = 0

    while i < context_len - n:
        if context_token_ids[i] == pattern[j]:
            i += 1
            j += 1
            if j == n:
                match_count += 1
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return match_count
