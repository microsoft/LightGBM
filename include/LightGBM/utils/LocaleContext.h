#ifndef __LOCALE_CONTEXT_H__
#define __LOCALE_CONTEXT_H__

#include <clocale>
#include <locale>

/**
 * Class to override the program global locale during this object lifetime.
 * After the object is destroyed, the locale is returned to its original state.
 *
 * @warn This is not thread-safe.
*/
class LocaleContext
{
public:

  /**
   * Override the current program global locale during this object lifetime.
   *
   * @param target_locale override the locale to this locale setting.
   * @warn This is not thread-safe.
   * @note This doesn't override cout, cerr, etc.
   */
  LocaleContext(const char* target_locale = "C") {
    std::locale::global(std::locale(target_locale));
  }

  /**
   * Restores the old global locale.
   */
  ~LocaleContext() {
    std::locale::global(_saved_global_locale);
  }
private:
  std::locale _saved_global_locale; //!< Stores global locale at initialization.
};

#endif // __LOCALE_CONTEXT_H__
