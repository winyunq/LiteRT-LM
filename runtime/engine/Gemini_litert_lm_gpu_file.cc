// Copyright 2025 The ODML Authors.
// Specialized tool for File-based GPU Stress Test & Benchmarking

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, backend, "gpu", "Backend (cpu, gpu)");
ABSL_FLAG(std::string, model_path, "", "Path to .litertlm file");
ABSL_FLAG(std::string, input_file, "", "Path to input text file for stress test");

namespace {
using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::EngineSettings;
using ::litert::lm::ModelAssets;
using ::nlohmann::json;

absl::Status RunStressTest(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  const std::string input_file = absl::GetFlag(FLAGS_input_file);
  
  if (model_path.empty() || input_file.empty()) {
      return absl::InvalidArgumentError("Both --model_path and --input_file are required.");
  }

  // 1. Read input file
  std::ifstream ifs(input_file);
  if (!ifs.is_open()) return absl::NotFoundError("Could not open input file.");
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string prompt = ss.str();

  // 2. Setup Engine
  ASSIGN_OR_RETURN(ModelAssets model_assets, ModelAssets::Create(model_path));
  ASSIGN_OR_RETURN(Backend backend, litert::lm::GetBackendFromString(absl::GetFlag(FLAGS_backend)));
  ASSIGN_OR_RETURN(EngineSettings engine_settings, EngineSettings::CreateDefault(std::move(model_assets), backend));
  engine_settings.GetMutableBenchmarkParams() = litert::lm::proto::BenchmarkParams();

  ASSIGN_OR_RETURN(auto engine, litert::lm::EngineFactory::CreateAny(std::move(engine_settings)));

  // 3. Setup Conversation
  ASSIGN_OR_RETURN(auto conv_config, ConversationConfig::Builder().SetSessionConfig(litert::lm::SessionConfig::CreateDefault()).Build(*engine));
  ASSIGN_OR_RETURN(auto conversation, Conversation::Create(*engine, conv_config));

  std::cout << "[System]: Starting Stress Test with file: " << input_file << std::endl;
  std::cout << "[System]: Input size: " << prompt.length() << " characters." << std::endl;

  // 4. Send Message (Sync for simplicity in benchmark)
  json content = {{"role", "user"}, {"content", {{{"type", "text"}, {"text", prompt}}}}};
  
  // Use a simple callback to see the output stream
  auto callback = [](absl::StatusOr<litert::lm::Message> msg) {
      if (msg.ok() && !msg->is_null()) {
          for (const auto& c : (*msg)["content"]) std::cout << c["text"].get<std::string>();
          std::cout << std::flush;
      }
  };

  RETURN_IF_ERROR(conversation->SendMessageAsync(content, callback));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Hours(1)));
  std::cout << std::endl;

  // 5. Output Metrics
  auto metrics = conversation->GetBenchmarkInfo();
  std::cout << "\n=======================================================" << std::endl;
  std::cout << " PERFORMANCE REPORT" << std::endl;
  std::cout << "=======================================================" << std::endl;
  std::cout << *metrics << std::endl;

  return absl::OkStatus();
}
}

int main(int argc, char** argv) {
  ABSL_CHECK_OK(RunStressTest(argc, argv));
  return 0;
}
