#pragma once

#ifndef FREEFORM2_TOKENIZER_H
#define FREEFORM2_TOKENIZER_H

#include <basic_types.h>
#include <list>
#include <map>
#include <string>
#include <vector>

namespace FreeForm2
{
    enum Token
    {
        // End-of-stream token.
        TOKEN_END,

        // Open parenthesis.
        TOKEN_OPEN,

        // Close parenthesis.
        TOKEN_CLOSE,

        // Open array.
        TOKEN_OPEN_ARRAY,

        // Close array.
        TOKEN_CLOSE_ARRAY,

        // An atom (a name).
        TOKEN_ATOM,

        // Integer.
        TOKEN_INT,

        // Floating point number.
        TOKEN_FLOAT,
    };

    // Class that turns a stream of characters into tokens.
    class Tokenizer
    {
    public:
        // Construct a tokeniser over the given input.
        explicit Tokenizer(SIZED_STRING p_input);

        // Advance to the next token, which is returned.
        Token Advance();

        // Get the current token type.
        Token GetToken() const;

        // Get the text that produced the current token.
        SIZED_STRING GetValue() const;

        // Gets the offset of the current token with respect to
        // the input.
        unsigned int GetPosition() const;

        // Returns the name of the given token.
        static const char* TokenName(Token p_token);

        // Signal the tokenizer to start recording tokens to be associated
        // with a macro. The current token is included in the macro.
        void StartMacro(SIZED_STRING p_macroName);

        // Signal the tokenizer that macro recording should end. The current
        // token will not be included in the macro. All subsequent ATOM tokens
        // will be compared against the macro name. Any matching atom will be 
        // expanded into the recorded macro tokens. Macros will be expanded 
        // during playback, not during recording.
        void EndMacro();

        // Delete a macro previously recorded with calls to Start/EndMacro.
        // This method returns true if the macro with the specified name was
        // successfully deleted; otherwise, returns false.
        bool DeleteMacro(SIZED_STRING p_macroName);

        // Return a flag to determine if the tokenizer is currently expanding
        // a macro.
        bool IsExpandingMacro() const;

        // Return a flag to determine if the tokenizer is currently recording
        // a macro.
        bool IsRecordingMacro() const;

    private:
        // Consume one character from the input.
        void AdvanceChar();

        // Advance the token being read from the input stream.
        Token AdvanceInput();

        // The original input.
        SIZED_STRING m_originalInput;

        // Remaining input.
        SIZED_STRING m_input;

        // Current token type.
        Token m_current;

        // Text that produced the current token.
        SIZED_STRING m_value;

        // This struct contains state data related to the recording of macros.
        // The macro state object can essentially be in one of three modes:
        // 1. No action is needed by the state.
        // 2. Macro expansion is in progress. The expansion of a macro is 
        //    referred to as 'playback' in this class.
        // 3. A macro is being recorded.
        struct MacroState
        {
            // Constructor to initialize flags.
            MacroState();

            // This method manages the macro recording state to start recording
            // a macro.
            void StartMacro(SIZED_STRING p_macroName);

            // Record a token to the current macro.
            void RecordToken(Token p_token, SIZED_STRING p_value);

            // This method manages the macro recording state to end recording a
            // macro. It also pops the last token off the stream to exclude it
            // from the macro.
            void EndMacro();

            // Signal if macro recording is in progress.
            bool IsRecording() const;

            // Signal if the macro playback is in progress.
            bool IsInPlayback() const;

            // A token recorded during macro recording.
            struct RecordedToken
            {
                // The type of the recorded token.
                Token m_token;

                // The string associated with the token.
                SIZED_STRING m_value;
            };

            // Return the current token. If not in playback, this will return
            // a TOKEN_END.
            RecordedToken GetCurrentToken() const;

            // Start the playback of a macro. Playback here means the stateful
            // expansion of a macro. If a macro does not exist for the given 
            // name, this function returns false.
            bool PlayMacro(SIZED_STRING p_name);

            // Advance playback by one token.
            void Advance();

            // Delete a macro by name. This function returns true if the macro
            // was successfully deleted; otherwise, returns false.
            bool DeleteMacro(SIZED_STRING p_name);

        private:
            // This type represents a stream of recorded tokens which can be
            // used to play back a macro.
            typedef std::vector<RecordedToken> MacroStream;

            // This type is the iterator type for macro streams.
            typedef MacroStream::const_iterator MacroStreamIter;

            // This type represents the playback state of a macro stream. The
            // beginning of the range is the current playback location, which
            // is advanced during playback until the range is empty.
            typedef std::pair<const MacroStream*, MacroStreamIter> PlaybackState;

            // A comparison functor to check less-than equality for 
            // SIZED_STRING objects.
            struct SizedStringLess
            {
                bool operator()(SIZED_STRING p_left, SIZED_STRING p_right) const;
            };

            // A map of macro names to the token stream with which they were 
            // recorded.
            std::map<SIZED_STRING, MacroStream, SizedStringLess> m_macros;

            // The stream to which macro recording is currently writing.
            MacroStream* m_recordingStream;

            // This list contains all playback states currently in progress.
            // The back element is the state of the macro currently being
            // expanded.
            std::list<PlaybackState> m_playbackStack;
        };

        MacroState m_macroState;
    };
}

#endif


