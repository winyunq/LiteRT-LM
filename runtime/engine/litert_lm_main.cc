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

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/internal/scoped_file.h"  // from @litert
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/status_macros.h"
#include "runtime/engine/shared_flags.h"

ABSL_FLAG(std::string, backend, "gpu", "Executor backend to use (cpu, gpu).");
ABSL_FLAG(std::string, model_path, "", "Model path.");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::nlohmann::json;

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateMessageCallback() {
  return [](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      std::cout << "Error: " << message.status() << std::endl;
      return;
    }
    if (message->is_null()) {
      std::cout << std::endl << std::flush;
      return;
    }
    for (const auto& content : (*message)["content"]) {
      std::cout << content["text"].get<std::string>();
    }
    std::cout << std::flush;
  };
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("Model path is empty.");
  }
  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,  // NOLINT
                        ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ABSL_ASSIGN_OR_RETURN(Backend backend,
                        litert::lm::GetBackendFromString(backend_str));
  ABSL_ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));
  // Enable benchmark by default.
  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();
  const int visual_token_budget = absl::GetFlag(FLAGS_visual_token_budget);
  if (visual_token_budget > 0) {
    engine_settings.SetMaxVisionTokensPerImage(visual_token_budget);
  }
  ABSL_ASSIGN_OR_RETURN(auto use_ir_model,
                        litert::lm::GetUseIrModelFromString(
                            absl::GetFlag(FLAGS_gpu_use_ir_model)));
  engine_settings.GetMutableMainExecutorSettings().SetUseIrModel(use_ir_model);

  if (backend == Backend::CPU) {
    auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
    ABSL_ASSIGN_OR_RETURN(
        auto cpu_settings,
        executor_settings.MutableBackendConfig<litert::lm::CpuConfig>());
    const int num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
    if (num_cpu_threads > 0) {
      cpu_settings.number_of_threads = num_cpu_threads;
    }
    cpu_settings.enable_ynnpack = absl::GetFlag(FLAGS_enable_ynnpack);
    executor_settings.SetBackendConfig(cpu_settings);
  }

  // Create the engine.
  ABSL_ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                         std::move(engine_settings)));

  auto session_config = litert::lm::SessionConfig::CreateDefault();
  ABSL_ASSIGN_OR_RETURN(auto conversation_config,
                        ConversationConfig::Builder()
                            .SetSessionConfig(session_config)
                            .Build(*engine));
  ABSL_ASSIGN_OR_RETURN(auto conversation,
                        Conversation::Create(*engine, conversation_config));

  std::cout << "=======================================================" << std::endl;
  std::cout << " LiteRT-LM Interactive Chat (" << backend_str << " Mode)" << std::endl;
  std::cout << " Type 'exit' to quit." << std::endl;
  std::cout << "=======================================================" << std::endl;

  while (true) {
    std::cout << "\nUser >> " << std::flush;
    std::string input;
    if (!std::getline(std::cin, input) || input == "exit") break;
    if (input.empty()) continue;
    std::cout << "\nModel >> " << std::flush;
    ABSL_RETURN_IF_ERROR(conversation->SendMessageAsync(
        json::object({{"role", "user"}, {"content", {{{"type", "text"}, {"text", input}}}}}),
        CreateMessageCallback()));
    ABSL_RETURN_IF_ERROR(engine->WaitUntilDone(absl::Hours(1)));
    std::cout << std::endl;
  }
  return absl::OkStatus();
}
}

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
