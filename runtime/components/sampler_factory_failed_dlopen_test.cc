// Copyright 2025 The ODML Authors.
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

#include <memory>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/sampler_factory.h"
#include "runtime/components/top_p_cpu_sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

TEST(SamplerFactoryFailedDlopenTest,
     CreateSamplerForGpuNeverFallsBackToCpuIfUnavailable) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto litert_env,
                               litert::Environment::Create({}));

  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  auto result =
      CreateSampler(Backend::GPU, /*batch_size=*/1, std::move(sampler_params),
                    litert_env.Get(), /*sequence_size=*/1, /*vocab_size=*/201,
                    /*activation_data_type=*/std::nullopt);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_THAT(result.status().message(),
              testing::HasSubstr("CPU fallback is disabled"));
}

TEST(SamplerFactoryFailedDlopenTest,
     CreateSamplerForExplicitCpuStillWorksWhenGpuLibraryIsUnavailable) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.0);
  sampler_params.set_temperature(1.0);
  sampler_params.set_seed(12345);
  sampler_params.set_type(proto::SamplerParameters::TOP_P);

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1,
                    std::move(sampler_params), /*env=*/nullptr,
                    /*sequence_size=*/1));
  EXPECT_NE(dynamic_cast<TopPSampler*>(sampler.get()), nullptr);
}

}  // namespace
}  // namespace litert::lm
