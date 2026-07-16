// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/logits_processor/repetition_penalty_processor.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::testing::AllOf;
using ::testing::Combine;
using ::testing::Contains;
using ::testing::Field;
using ::testing::Key;
using ::testing::Not;
using ::testing::Pair;
using ::testing::TestWithParam;
using ::testing::UnorderedElementsAre;
using ::testing::Values;
using ::testing::status::StatusIs;

enum class ProcessLogitsFloatVariant {
  kSpanFloat,
  kSpanHalf,
};

class RepetitionPenaltyProcessorParamTest
    : public TestWithParam<ProcessLogitsFloatVariant> {
 protected:
  template <typename Func>
  void RunWithParam(Func&& func) {
    switch (GetParam()) {
      case ProcessLogitsFloatVariant::kSpanFloat:
        func(float{});
        break;
      case ProcessLogitsFloatVariant::kSpanHalf:
        func(tflite::half{});
        break;
    }
  }
};

TEST_P(RepetitionPenaltyProcessorParamTest, AppliesMultiplicativePenalty) {
  // Batch size = 1, Vocab size = 3
  // Repetition penalty = 1.2
  // Presence penalty = 0.0 (disabled)
  // Frequency penalty = 0.0 (disabled)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.2f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/3,
                                       config);

  // Update state with token 1.
  std::vector<int> tokens = {1};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 3};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    // Logits for tokens [0, 1, 2]
    std::vector<T> logits = {static_cast<T>(2.0f), static_cast<T>(1.2f),
                             static_cast<T>(-1.2f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Token 0 and 2 were not seen, so their logits remain unchanged.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]),
                    static_cast<float>(static_cast<T>(-1.2f)));

    // Token 1 was seen, its positive logit should be scaled down by 1.2
    // (1.2 / 1.2 = 1.0f)
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 1.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest,
       MultiplicativePenaltyScalesNegatives) {
  // Batch size = 1, Vocab size = 2
  // Repetition penalty = 1.5
  // Presence penalty = 0.0 (disabled)
  // Frequency penalty = 0.0 (disabled)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/2,
                                       config);

  // Update state with token 0 twice.
  std::vector<int> tokens = {0};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 2};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(-2.0f), static_cast<T>(3.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Token 0 was seen, its negative logit should be scaled UP by 1.5
    // (-2.0 * 1.5 = -3.0f)
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), -3.0f);

    // Token 1 was not seen.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 3.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest,
       AppliesPresenceAndFrequencyPenalties) {
  // Batch size = 1, Vocab size = 3
  // Repetition penalty = 1.0 (disabled)
  // Presence penalty = 2.0
  // Frequency penalty = 1.0 (per-token penalty)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.0f, /*presence_penalty=*/2.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/3,
                                       config);

  // See token 0 once, token 1 twice.
  std::vector<int> tok_0 = {0};
  std::vector<int> tok_1 = {1};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_0)));
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_1)));
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_1)));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 3};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f),
                             static_cast<T>(10.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Token 0 seen 1 time: Presence (2.0) + Freq (1 * 1.0) = -3.0.
    // Logit = 7.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 7.0f);

    // Token 1 seen 2 times: Presence (2.0) + Freq (2 * 1.0) = -4.0.
    // Logit = 6.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 6.0f);

    // Token 2 seen 0 times: Logit = 10.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 10.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest,
       AppliesMultiplicativeThenSubtractive) {
  // Batch size = 1, Vocab size = 1
  // Repetition penalty = 1.5
  // Presence penalty = 1.0
  // Frequency penalty = 0.0 (disabled)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/1,
                                       config);

  // Update state with token 0.
  std::vector<int> tokens = {0};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 1};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(3.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // If subtractive was first: (3.0 - 1.0) / 1.5 = 1.333
    // Multiplicative first: (3.0 / 1.5) - 1.0 = 2.0 - 1.0 = 1.0
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 1.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest,
       HandlesNegativePresenceAndFrequencyPenalties) {
  // Batch size = 1, Vocab size = 1
  // Repetition penalty = 3.0
  // Presence penalty = -1.0
  // Frequency penalty = -2.0
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/3.0f, /*presence_penalty=*/-1.0f,
      /*frequency_penalty=*/-2.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/1,
                                       config);

  // Update state with token 0.
  std::vector<int> tokens = {0};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 1};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(-3.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Repetition: -3.0 * 3.0 = -9.0
    // Presence: -1.0
    // Freq: 1 * -2.0 = -2.0.
    // Logit = -9.0 - (-1.0) - (-2.0) = -6.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), -6.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest, ObservesSlidingWindowLength) {
  // Batch size = 1, Vocab size = 3
  // Repetition penalty = 1.0 (disabled)
  // Presence penalty = 0.0 (disabled)
  // Frequency penalty = 1.0 (per-token penalty)
  // Repetition window size = 2
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.0f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/2);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/3,
                                       config);

  // Update state sequentially: [0], [1], [2].
  // The first token [0] should fall out of the size-2 window window tracking.
  std::vector<int> tok_0 = {0};
  std::vector<int> tok_1 = {1};
  std::vector<int> tok_2 = {2};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_0)));
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_1)));
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tok_2)));

  // Token 0 is no longer in the window, while tokens 1 and 2 are in the window
  // once.
  EXPECT_THAT(
      processor.GetBatchStatesForTesting(),
      ElementsAre(Field(&RepetitionPenaltyProcessor::BatchState::token_counts,
                        AllOf(Not(Contains(Key(0))),
                              UnorderedElementsAre(Pair(1, 1), Pair(2, 1))))));

  std::vector<::litert::Layout::Dim> dims = {1, 1, 3};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f),
                             static_cast<T>(10.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Token 0 fell off window, penalty zero. Logit = 10.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 10.0f);

    // Token 1 in window (once). Freq (1 * 1.0) = -1.0. Logit = 9.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 9.0f);

    // Token 2 in window (once). Freq (1 * 1.0) = -1.0. Logit = 9.0f.
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 9.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest, HandlesBatchedSequences) {
  // Batch size = 2, Vocab size = 2
  // Repetition penalty = 1.5
  // Presence penalty = 0.0 (disabled)
  // Frequency penalty = 0.0 (disabled)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/1.5f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/2, /*vocab_size=*/2,
                                       config);

  // Sequence 0 sees token 1. Sequence 1 sees token 0.
  std::vector<int> tokens = {1, 0};
  EXPECT_OK(processor.UpdateState(absl::MakeSpan(tokens)));

  std::vector<::litert::Layout::Dim> dims = {2, 1, 2};

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    // Layout: [Seq 0 (vocab 0, vocab 1), Seq 1 (vocab 0, vocab 1)]
    std::vector<T> logits = {static_cast<T>(3.0f), static_cast<T>(3.0f),
                             static_cast<T>(3.0f), static_cast<T>(3.0f)};

    EXPECT_OK(
        processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)));

    // Seq 0 Token 0 (not seen) -> 3.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[0]), 3.0f);
    // Seq 0 Token 1 (seen) -> 3.0f / 1.5f = 2.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[1]), 2.0f);

    // Seq 1 Token 0 (seen) -> 3.0f / 1.5f = 2.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[2]), 2.0f);
    // Seq 1 Token 1 (not seen) -> 3.0f
    EXPECT_FLOAT_EQ(static_cast<float>(logits[3]), 3.0f);
  });
}

TEST_P(RepetitionPenaltyProcessorParamTest, HandlesTensorBuffer) {
  // Batch size = 1, Vocab size = 2
  // Repetition penalty = 2.0
  // Presence penalty = 0.0 (disabled)
  // Frequency penalty = 0.0 (disabled)
  // Repetition window size = 0 (track all tokens)
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/0.0f,
      /*frequency_penalty=*/0.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/1, /*vocab_size=*/2,
                                       config);

  std::vector<int> tokens = {0};
  auto tb_tokens = CopyToTensorBuffer<int>(absl::MakeSpan(tokens),
                                           /*dims=*/{1, 1});
  ASSERT_TRUE(tb_tokens.HasValue());
  EXPECT_OK(processor.UpdateState(*tb_tokens));

  RunWithParam([&](auto dummy_type) {
    using T = decltype(dummy_type);
    std::vector<T> logits = {static_cast<T>(10.0f), static_cast<T>(10.0f)};
    ::litert::Dimensions tb_dims = {1, 1, 2};
    auto tb_result =
        CopyToTensorBuffer<T>(absl::MakeSpan(logits), std::move(tb_dims));
    ASSERT_TRUE(tb_result.HasValue());

    EXPECT_OK(processor.ProcessLogits(*tb_result));

    auto modified_logits = CopyFromTensorBuffer<T>(*tb_result);
    ASSERT_TRUE(modified_logits.HasValue());
    EXPECT_FLOAT_EQ(static_cast<float>((*modified_logits)[0]), 5.0f);
    EXPECT_FLOAT_EQ(static_cast<float>((*modified_logits)[1]), 10.0f);
  });
}

INSTANTIATE_TEST_SUITE_P(ProcessLogitsVariant,
                         RepetitionPenaltyProcessorParamTest,
                         Values(ProcessLogitsFloatVariant::kSpanFloat,
                                ProcessLogitsFloatVariant::kSpanHalf));

class RepetitionPenaltyProcessorInvalidDimsTest
    : public TestWithParam<std::tuple<ProcessLogitsFloatVariant,
                                      std::vector<::litert::Layout::Dim>>> {};

TEST_P(RepetitionPenaltyProcessorInvalidDimsTest,
       ReturnsErrorForInvalidLogitsDims) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/2, /*vocab_size=*/5,
                                       config);

  std::vector<float> logits_float(10, 1.0f);
  std::vector<tflite::half> logits_half(10, static_cast<tflite::half>(1.0f));

  ProcessLogitsFloatVariant variant = std::get<0>(GetParam());
  std::vector<::litert::Layout::Dim> dims = std::get<1>(GetParam());

  switch (variant) {
    case ProcessLogitsFloatVariant::kSpanFloat:
      EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_float),
                                          absl::MakeSpan(dims)),
                  StatusIs(absl::StatusCode::kInvalidArgument));
      break;
    case ProcessLogitsFloatVariant::kSpanHalf:
      EXPECT_THAT(processor.ProcessLogits(absl::MakeSpan(logits_half),
                                          absl::MakeSpan(dims)),
                  StatusIs(absl::StatusCode::kInvalidArgument));
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(
    InvalidDims, RepetitionPenaltyProcessorInvalidDimsTest,
    Combine(
        Values(ProcessLogitsFloatVariant::kSpanFloat,
               ProcessLogitsFloatVariant::kSpanHalf),
        Values(
            std::vector<::litert::Layout::Dim>{
                2, 5},  // Invalid size (2D instead of 3D)
            std::vector<::litert::Layout::Dim>{1, 1, 5},  // Invalid batch size
            std::vector<::litert::Layout::Dim>{
                2, 2, 5},  // Invalid sequence length (must be 1)
            std::vector<::litert::Layout::Dim>{2, 1, 4}  // Invalid vocab size
            )));

TEST(RepetitionPenaltyProcessorTest,
     ReturnsErrorForInvalidTensorBufferElementType) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/2, /*vocab_size=*/5,
                                       config);

  std::vector<::litert::Layout::Dim> tokens_dims = {2, 1};
  std::vector<::litert::Layout::Dim> logits_dims = {2, 1, 5};

  int num_tokens = 1;
  for (int dim : tokens_dims) {
    num_tokens *= dim;
  }
  int num_logits = 1;
  for (int dim : logits_dims) {
    num_logits *= dim;
  }

  // UpdateState with float TensorBuffer should fail.
  std::vector<float> tokens_float(num_tokens, 1.5f);
  auto tb_tokens_float = CopyToTensorBuffer<float>(
      absl::MakeSpan(tokens_float),
      ::litert::Dimensions(tokens_dims.begin(), tokens_dims.end()));
  ASSERT_TRUE(tb_tokens_float.HasValue());
  EXPECT_FALSE(processor.UpdateState(*tb_tokens_float).ok());

  // UpdateState with half TensorBuffer should fail.
  std::vector<tflite::half> tokens_half(num_tokens,
                                        static_cast<tflite::half>(1.5f));
  auto tb_tokens_half = CopyToTensorBuffer<tflite::half>(
      absl::MakeSpan(tokens_half),
      ::litert::Dimensions(tokens_dims.begin(), tokens_dims.end()));
  ASSERT_TRUE(tb_tokens_half.HasValue());
  EXPECT_FALSE(processor.UpdateState(*tb_tokens_float).ok());

  // ProcessLogits with Int32 TensorBuffer should fail.
  std::vector<int32_t> logits_int32(num_logits, 1);
  auto tb_logits_int32 = CopyToTensorBuffer<int32_t>(
      absl::MakeSpan(logits_int32),
      ::litert::Dimensions(logits_dims.begin(), logits_dims.end()));
  ASSERT_TRUE(tb_logits_int32.HasValue());
  EXPECT_THAT(processor.ProcessLogits(*tb_logits_int32),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported logits tensor format element type."));

  // ProcessLogits with Int8 TensorBuffer should fail.
  std::vector<int8_t> logits_int8(num_logits, 1);
  auto tb_int8_result = CopyToTensorBuffer<int8_t>(
      absl::MakeSpan(logits_int8),
      ::litert::Dimensions(logits_dims.begin(), logits_dims.end()));
  ASSERT_TRUE(tb_int8_result.HasValue());
  EXPECT_THAT(processor.ProcessLogits(*tb_int8_result),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported logits tensor format element type."));
}

TEST(RepetitionPenaltyProcessorTest, ReturnsErrorForInvalidLogitsSize) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/2, /*vocab_size=*/5,
                                       config);

  std::vector<::litert::Layout::Dim> dims = {2, 1, 5};

  // Mis-sized logits
  std::vector<float> logits(11, 1.0f);
  EXPECT_THAT(
      processor.ProcessLogits(absl::MakeSpan(logits), absl::MakeSpan(dims)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Logits span size incorrectly mapped."));
}

TEST(RepetitionPenaltyProcessorTest, ReturnsErrorForInvalidUpdateState) {
  RepetitionPenaltyConfig config(
      /*repetition_penalty=*/2.0f, /*presence_penalty=*/1.0f,
      /*frequency_penalty=*/1.0f, /*window_size=*/0);
  RepetitionPenaltyProcessor processor(/*batch_size=*/2, /*vocab_size=*/5,
                                       config);

  // Wrong batch size
  std::vector<int> tokens_wrong_size = {1};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_wrong_size)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid token ID (-1)
  std::vector<int> tokens_invalid_negative = {1, -1};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_invalid_negative)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid token ID (vocab_size)
  std::vector<int> tokens_invalid_large = {1, 5};
  EXPECT_THAT(processor.UpdateState(absl::MakeSpan(tokens_invalid_large)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
