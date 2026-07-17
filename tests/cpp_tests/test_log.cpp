/*!
 * Copyright (c) 2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>
#include <LightGBM/utils/log.h>

#include <stdexcept>
#include <string>

using LightGBM::Log;
using LightGBM::LogLevel;

namespace {

struct CaptureState {
  int leveled_call_count = 0;
  int last_level = 0;
  std::string last_message;
  int legacy_call_count = 0;
};

// Function-local static rather than a namespace-scope global, to satisfy cpplint's
// runtime/string check (no non-POD globals); same pattern as Log's private accessors.
CaptureState& GetCaptureState() {
  static CaptureState state;
  return state;
}

void LeveledCaptureCallback(int level, const char* msg) {
  CaptureState& state = GetCaptureState();
  ++state.leveled_call_count;
  state.last_level = level;
  state.last_message = msg;
}

void LegacyCaptureCallback(const char* msg) {
  (void)msg;
  ++GetCaptureState().legacy_call_count;
}

}  // namespace

class LogCallbackTest : public testing::Test {
 public:
  void SetUp() override {
    CaptureState& state = GetCaptureState();
    state.leveled_call_count = 0;
    state.last_level = 0;
    state.last_message.clear();
    state.legacy_call_count = 0;
    // let Debug-level messages reach the callback
    Log::ResetLogLevel(LogLevel::Debug);
  }

  void TearDown() override {
    // clear both thread-local callbacks so later tests get default logging
    Log::ResetCallBackWithLevel(nullptr);
    Log::ResetCallBack(nullptr);
    Log::ResetLogLevel(LogLevel::Info);  // restore default
  }
};

TEST_F(LogCallbackTest, LeveledCallbackReceivesOneCallPerMessage) {
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);

  Log::Info("hello %d", 42);

  const CaptureState& state = GetCaptureState();
  EXPECT_EQ(1, state.leveled_call_count);
  EXPECT_EQ(static_cast<int>(LogLevel::Info), state.last_level);
  EXPECT_EQ("hello 42", state.last_message);
}

TEST_F(LogCallbackTest, LeveledCallbackTakesPrecedenceOverLegacyCallback) {
  Log::ResetCallBack(LegacyCaptureCallback);
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);

  Log::Warning("legacy should be shadowed");

  const CaptureState& state = GetCaptureState();
  EXPECT_EQ(1, state.leveled_call_count);
  EXPECT_EQ(0, state.legacy_call_count);
  EXPECT_EQ(static_cast<int>(LogLevel::Warning), state.last_level);
}

TEST_F(LogCallbackTest, FatalInvokesLeveledCallbackWithFatalLevelThenThrows) {
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);

  EXPECT_THROW(Log::Fatal("fatal message %s", "here"), std::runtime_error);

  const CaptureState& state = GetCaptureState();
  EXPECT_EQ(1, state.leveled_call_count);
  EXPECT_EQ(static_cast<int>(LogLevel::Fatal), state.last_level);
  EXPECT_EQ("fatal message here", state.last_message);
}

TEST_F(LogCallbackTest, ResettingToNullptrStopsFurtherInvocations) {
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);
  Log::Info("first");
  EXPECT_EQ(1, GetCaptureState().leveled_call_count);

  Log::ResetCallBackWithLevel(nullptr);
  Log::Info("second");  // goes to stdout; the callback must not fire
  EXPECT_EQ(1, GetCaptureState().leveled_call_count);
}

TEST_F(LogCallbackTest, MessagesBelowActiveLevelNeverReachLeveledCallback) {
  // override SetUp's Debug verbosity: at Info, Debug messages are filtered before the callback
  Log::ResetLogLevel(LogLevel::Info);
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);

  Log::Debug("filtered out");
  EXPECT_EQ(0, GetCaptureState().leveled_call_count);

  Log::Info("passes through");
  EXPECT_EQ(1, GetCaptureState().leveled_call_count);
  EXPECT_EQ(static_cast<int>(LogLevel::Info), GetCaptureState().last_level);
}

TEST_F(LogCallbackTest, FatalSuppressesStderrWhenLeveledCallbackIsSet) {
  Log::ResetCallBackWithLevel(LeveledCaptureCallback);

  testing::internal::CaptureStderr();
  EXPECT_THROW(Log::Fatal("fatal message %s", "here"), std::runtime_error);
  const std::string captured_stderr = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(captured_stderr.empty()) << "stderr should be suppressed when a leveled callback "
                                           "is registered, but got: "
                                        << captured_stderr;
  EXPECT_EQ(1, GetCaptureState().leveled_call_count);
}
