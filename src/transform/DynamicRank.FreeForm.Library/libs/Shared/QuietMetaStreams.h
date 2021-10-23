#pragma once

#include <cstdio>
#include "FreeForm2Assert.h"
#include <io.h>
#include <memory>
#include <MetaStreams.h>

namespace FreeForm2
{
    // This class redirects an output file to a NUL output stream and restores
    // the original output mechanism when destroyed. If the class fails to 
    // restore the output stream for some reason, it is redirected to console
    // output.
    class IORedirectGuard
    {
    public:
        static const int c_badFileDescriptor = -1;

        // Redirect the file parameter to a NUL output stream. Do nothing if 
        // any I/O call fails.
        explicit IORedirectGuard(FILE* p_out)
            : m_out(p_out),
              m_redirect(nullptr),
              m_duplicatedFd(c_badFileDescriptor)
        {
            m_redirect = fopen("NUL", "w");
            if (m_redirect)
            {
                m_duplicatedFd = _dup(_fileno(m_out));
                if (m_duplicatedFd == c_badFileDescriptor)
                {
                    // On failure, close the redirect file handle.
                    fclose(m_redirect);
                    m_redirect = nullptr;
                }
                else
                {
                    if (_dup2(_fileno(m_redirect), _fileno(m_out)) != 0)
                    {
                        // On failure, close the opened handles.
                        fclose(m_redirect);
                        m_redirect = nullptr;
                        _close(m_duplicatedFd);
                        m_duplicatedFd = c_badFileDescriptor;
                    }
                }
            }
        }

        // Restore the output file descriptor.
        ~IORedirectGuard()
        {
            if (m_redirect)
            {
                fflush(m_out);
                fclose(m_redirect);
                if (_dup2(m_duplicatedFd, _fileno(m_out)) != 0)
                {
                    // On failure, redirect the output stream to console 
                    // output. This could cause issues with scripts, but it's
                    // better than silencing the output.
                    FILE* out = nullptr;
                    const errno_t err = freopen_s(&out, "CONOUT$", "w", m_out);
                    FF2_ASSERT(err == 0);
                }
            }
        }

    private:
        // The file being redirected.
        FILE* m_out;

        // The newly created output stream to which m_out is redirected.
        FILE* m_redirect;

        // The duplicated file descriptor of the original output source.
        int m_duplicatedFd;
    };


    // Load a metastream definition list from a file, returning a unique_ptr
    // to the object. Because the MetaStreams constructor produces (incredibly)
    // versbose output, this method will silence stdout and stderr during the
    // loading process.
    template<typename String>
    std::unique_ptr<MetaStreams> QuietLoadMSDL(const String& p_path)
    {
        std::unique_ptr<MetaStreams> msdl;
        {
            IORedirectGuard stdOutGuard(stdout);
            IORedirectGuard stdErrGuard(stderr);
            msdl.reset(new MetaStreams(p_path));
        }
        return msdl;
    }
}