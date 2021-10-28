/*!
 * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#include "FreeForm2Tokenizer.h"

#include <ctype.h>

#include <sstream>
#include <stdexcept>

#include "FreeForm2Assert.h"

FreeForm2::Tokenizer::Tokenizer(SIZED_STRING p_input)
    : m_input(p_input), m_originalInput(p_input) {
  Advance();
}

FreeForm2::Token FreeForm2::Tokenizer::GetToken() const { return m_current; }

FreeForm2::Token FreeForm2::Tokenizer::Advance() {
  // The Tokenizer should never be both expanding and recording a macro at
  // the same time.
  FF2_ASSERT(!(m_macroState.IsInPlayback() && m_macroState.IsRecording()));

  // Check for macro playback first.
  if (m_macroState.IsInPlayback()) {
    MacroState::RecordedToken token = m_macroState.GetCurrentToken();
    m_value = token.m_value;
    m_current = token.m_token;
  } else {
    Token token = AdvanceInput();
    FF2_ASSERT(m_current == token);
  }

  if (m_macroState.IsRecording()) {
    m_macroState.RecordToken(m_current, m_value);
  } else {
    // Check for macro expansion.
    while (m_current == TOKEN_ATOM && m_macroState.PlayMacro(m_value)) {
      MacroState::RecordedToken token = m_macroState.GetCurrentToken();
      m_current = token.m_token;
      m_value = token.m_value;
    }
  }

  if (m_macroState.IsInPlayback()) {
    m_macroState.Advance();
  }

  return m_current;
}

FreeForm2::Token FreeForm2::Tokenizer::AdvanceInput() {
  // Remove all comments and whitespace from the front of the input.  Note
  // that we have to loop, as comments and whitespace can be arbitrarily long.
  bool reduced = false;
  do {
    reduced = false;

    // Discard comments.
    if (m_input.cbData > 0 && m_input.pbData[0] == '#') {
      reduced = true;
      AdvanceChar();
      while (m_input.cbData > 0 && m_input.pcData[0] != '\r' &&
             m_input.pcData[0] != '\n') {
        AdvanceChar();
      }
    }

    // Discard whitespace.
    while (m_input.cbData > 0 && isspace(m_input.pbData[0])) {
      reduced = true;
      AdvanceChar();
    }
  } while (reduced);

  if (m_input.cbData == 0) {
    m_value.cbData = 0;
    return (m_current = TOKEN_END);
  }

  if (m_input.pbData[0] == '(') {
    m_value = m_input;
    m_value.cbData = 1;
    AdvanceChar();
    return (m_current = TOKEN_OPEN);
  } else if (m_input.pbData[0] == ')') {
    m_value = m_input;
    m_value.cbData = 1;
    AdvanceChar();
    return (m_current = TOKEN_CLOSE);
  } else if (m_input.pbData[0] == '[') {
    m_value = m_input;
    m_value.cbData = 1;
    AdvanceChar();
    return (m_current = TOKEN_OPEN_ARRAY);
  } else if (m_input.pbData[0] == ']') {
    m_value = m_input;
    m_value.cbData = 1;
    AdvanceChar();
    return (m_current = TOKEN_CLOSE_ARRAY);
  } else if (isdigit(m_input.pbData[0]) ||
             (m_input.pbData[0] == '-' && m_input.cbData > 1 &&
              isdigit(m_input.pbData[1]))) {
    // Parse literal numeric value.  Note that we had to use a lookahead to
    // tell the difference between '-1.0' and '-' (the atom).
    m_value.pbData = m_input.pbData;
    AdvanceChar();

    while (m_input.cbData > 0 && isdigit(m_input.pbData[0])) {
      AdvanceChar();
    }

    Token tok = TOKEN_INT;

    // Parse decimal in float.
    if (m_input.cbData > 0 && m_input.pcData[0] == '.') {
      tok = TOKEN_FLOAT;
      AdvanceChar();

      while (m_input.cbData > 0 && isdigit(m_input.pbData[0])) {
        AdvanceChar();
      }
    }

    // Allow exponents on floating point numbers.
    if (m_input.cbData > 0 &&
        (m_input.pcData[0] == 'e' || m_input.pcData[0] == 'E')) {
      tok = TOKEN_FLOAT;
      AdvanceChar();

      if (m_input.cbData > 0 &&
          (m_input.pcData[0] == '-' || m_input.pcData[0] == '+')) {
        // Allow negative exponent.
        AdvanceChar();
      }

      while (m_input.cbData > 0 && isdigit(m_input.pbData[0])) {
        AdvanceChar();
      }
    }

    m_value.cbData = m_input.pbData - m_value.pbData;
    return (m_current = tok);
  } else if (isalpha(m_input.pbData[0])) {
    // Parse atom.
    m_value.pbData = m_input.pbData;
    AdvanceChar();

    while (m_input.cbData > 0 &&
               (isalnum(m_input.pbData[0]) || m_input.pbData[0] == '_' ||
                m_input.pbData[0] == '-' || m_input.pbData[0] == ':' ||
                m_input.pbData[0] == '@') ||
           m_input.pbData[0] == '.' || m_input.pbData[0] == '|') {
      AdvanceChar();
    }

    m_value.cbData = m_input.pbData - m_value.pbData;
    return (m_current = TOKEN_ATOM);
  } else if (ispunct(m_input.pbData[0]) && m_input.pcData[0] != '#' &&
             m_input.pcData[0] != '@') {
    // Parse atom.
    m_value.pbData = m_input.pbData;

    for (AdvanceChar(); m_input.cbData > 0 && ispunct(m_input.pbData[0]) &&
                        m_input.pbData[0] != '#' && m_input.pbData[0] != '@' &&
                        m_input.pbData[0] != ')' && m_input.pbData[0] != ']' &&
                        m_input.pbData[0] != '(' && m_input.pbData[0] != '[';
         AdvanceChar())
      ;

    m_value.cbData = m_input.pbData - m_value.pbData;
    return (m_current = TOKEN_ATOM);
  } else {
    std::ostringstream err;
    err << "Invalid character '" << m_input.pcData[0] << "' (ascii "
        << static_cast<unsigned int>(m_input.pbData[0])
        << " in decimal) found in input.";
    throw std::runtime_error(err.str());
  }
}

SIZED_STRING
FreeForm2::Tokenizer::GetValue() const { return m_value; }

unsigned int FreeForm2::Tokenizer::GetPosition() const {
  return static_cast<unsigned int>(m_value.pbData - m_originalInput.pbData);
}

void FreeForm2::Tokenizer::AdvanceChar() {
  m_input.pbData++;
  m_input.cbData--;
}

const char *FreeForm2::Tokenizer::TokenName(Token p_token) {
  switch (p_token) {
    case TOKEN_END:
      return "end";
    case TOKEN_OPEN:
      return "open";
    case TOKEN_CLOSE:
      return "close";
    case TOKEN_OPEN_ARRAY:
      return "open array";
    case TOKEN_CLOSE_ARRAY:
      return "close array";
    case TOKEN_ATOM:
      return "atom";
    case TOKEN_INT:
      return "int";
    case TOKEN_FLOAT:
      return "float";

    default: {
      Unreachable(__FILE__, __LINE__);
      break;
    }
  }
}

void FreeForm2::Tokenizer::StartMacro(SIZED_STRING p_macroName) {
  if (m_macroState.IsInPlayback()) {
    std::ostringstream err;
    err << "Cannot define a macro while another macro is being expanded "
        << "(additional macro definition at offset " << GetPosition() << ")";
    throw std::runtime_error(err.str());
  }
  m_macroState.StartMacro(p_macroName);
}

void FreeForm2::Tokenizer::EndMacro() { m_macroState.EndMacro(); }

bool FreeForm2::Tokenizer::DeleteMacro(SIZED_STRING p_name) {
  return m_macroState.DeleteMacro(p_name);
}

bool FreeForm2::Tokenizer::IsExpandingMacro() const {
  return m_macroState.IsInPlayback();
}

bool FreeForm2::Tokenizer::IsRecordingMacro() const {
  return m_macroState.IsRecording();
}

FreeForm2::Tokenizer::MacroState::MacroState() : m_recordingStream(nullptr) {}

void FreeForm2::Tokenizer::MacroState::StartMacro(SIZED_STRING p_macroName) {
  FF2_ASSERT(!IsRecording() && "Cannot nest macro definitions");

  FF2_ASSERT(p_macroName.cbData > 0 && p_macroName.pcData != nullptr &&
             "Macro name cannot be empty");

  if (m_macros.find(p_macroName) != m_macros.end()) {
    std::ostringstream err;
    err << "Macro already defined: " << SIZED_STR_STL(p_macroName);
    throw std::runtime_error(err.str());
  }

  const auto ret = m_macros.insert(std::make_pair(p_macroName, MacroStream()));
  FF2_ASSERT(ret.second && ret.first != m_macros.end());
  m_recordingStream = &ret.first->second;
}

void FreeForm2::Tokenizer::MacroState::RecordToken(Token p_token,
                                                   SIZED_STRING p_value) {
  FF2_ASSERT(IsRecording() && "Macro recording not started");
  RecordedToken token;
  token.m_token = p_token;
  token.m_value = p_value;
  m_recordingStream->push_back(token);
}

void FreeForm2::Tokenizer::MacroState::EndMacro() {
  FF2_ASSERT(IsRecording() && "Macro recording not started");
  FF2_ASSERT(!m_recordingStream->empty() &&
             "Macro token stream cannot be empty");

  m_recordingStream->pop_back();

  m_recordingStream = nullptr;
}

bool FreeForm2::Tokenizer::MacroState::IsRecording() const {
  return m_recordingStream != nullptr;
}

bool FreeForm2::Tokenizer::MacroState::IsInPlayback() const {
  return !m_playbackStack.empty();
}

FreeForm2::Tokenizer::MacroState::RecordedToken
FreeForm2::Tokenizer::MacroState::GetCurrentToken() const {
  if (IsInPlayback()) {
    const PlaybackState &state = m_playbackStack.back();
    FF2_ASSERT(state.second != state.first->cend());
    return *state.second;
  } else {
    const RecordedToken token = {TOKEN_END, {nullptr, 0}};
    return token;
  }
}

bool FreeForm2::Tokenizer::MacroState::PlayMacro(SIZED_STRING p_name) {
  const auto find = m_macros.find(p_name);
  if (find != m_macros.end()) {
    const MacroStream &stream = find->second;
    PlaybackState range(&stream, stream.cbegin());
    FF2_ASSERT(!range.first->empty() && "Empty macros are not allowed.");

    for (auto iter = m_playbackStack.cbegin(); iter != m_playbackStack.cend();
         ++iter) {
      if (&stream == iter->first) {
        std::ostringstream err;
        err << "Macro definition is malformed: recursive macros are not "
               "allowed "
            << "for macro: " << SIZED_STR_STL(p_name);
        throw std::runtime_error(err.str());
      }
    }

    m_playbackStack.push_back(range);
    return true;
  } else {
    return false;
  }
}

void FreeForm2::Tokenizer::MacroState::Advance() {
  FF2_ASSERT(IsInPlayback() && "Must be in playback to advance token");

  PlaybackState &range = m_playbackStack.back();
  FF2_ASSERT(range.second != range.first->cend());
  ++range.second;

  if (range.second == range.first->cend()) {
    // This invalidates the reference contained by range. Range may not be
    // accessed after the pop call.
    m_playbackStack.pop_back();
    if (IsInPlayback()) {
      Advance();
    }
  }
}

bool FreeForm2::Tokenizer::MacroState::DeleteMacro(SIZED_STRING p_name) {
  const auto find = m_macros.find(p_name);
  if (find != m_macros.cend()) {
    const MacroStream &stream = find->second;
    for (auto iter = m_playbackStack.cbegin(); iter != m_playbackStack.cend();
         ++iter) {
      if (iter->first == &stream) {
        std::ostringstream err;
        err << "Macro definition is malformed or contains more closing "
            << "tokens than opening. Name: " << SIZED_STR_STL(p_name);
        throw std::runtime_error(err.str());
      }
    }

    m_macros.erase(find);
    return true;
  } else {
    return false;
  }
}

bool FreeForm2::Tokenizer::MacroState::SizedStringLess::operator()(
    SIZED_STRING p_left, SIZED_STRING p_right) const {
  if (p_left.cbData < p_right.cbData) {
    return std::char_traits<char>::compare(p_left.pcData, p_right.pcData,
                                           p_left.cbData) <= 0;
  } else {
    return std::char_traits<char>::compare(p_left.pcData, p_right.pcData,
                                           p_right.cbData) < 0;
  }
}
