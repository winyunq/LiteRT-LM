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

#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/globals.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "nlohmann/json.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

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
  if (model_path.empty()) return absl::InvalidArgumentError("Model path is empty.");
  
  ASSIGN_OR_RETURN(ModelAssets model_assets, ModelAssets::Create(model_path));
  auto backend_str = absl::GetFlag(FLAGS_backend);
  ASSIGN_OR_RETURN(Backend backend, litert::lm::GetBackendFromString(backend_str));
  ASSIGN_OR_RETURN(EngineSettings engine_settings, EngineSettings::CreateDefault(std::move(model_assets), backend));
  engine_settings.GetMutableBenchmarkParams() = litert::lm::proto::BenchmarkParams();

<<<<<<< HEAD
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateAny(std::move(engine_settings)));
=======
  // Create the engine.
  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateDefault(
                                    std::move(engine_settings)));
>>>>>>> upstream/main

  auto session_config = litert::lm::SessionConfig::CreateDefault();
  ASSIGN_OR_RETURN(auto conversation_config, ConversationConfig::Builder().SetSessionConfig(session_config).Build(*engine));
  ASSIGN_OR_RETURN(auto conversation, Conversation::Create(*engine, conversation_config));

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
    RETURN_IF_ERROR(conversation->SendMessageAsync(
        json::object({{"role", "user"}, {"content", {{{"type", "text"}, {"text", input}}}}}),
        CreateMessageCallback()));
    RETURN_IF_ERROR(engine->WaitUntilDone(absl::Hours(1)));
    std::cout << std::endl;
  }
  return absl::OkStatus();
}
}

int main(int argc, char** argv) {
  ABSL_CHECK_OK(MainHelper(argc, argv));
  return 0;
}
