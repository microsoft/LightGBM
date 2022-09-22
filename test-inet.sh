echo "------ testing test program ----"

cat > conftest.cpp <<EOL
#include <ws2tcpip.h>
int main() {
  void (*fptr)(int, const char*, void*);
  fptr = &inet_pton;
  return 0;
}
EOL

g++ -std=gnu++11 -I"c:/rtools42/x86_64-w64-mingw32.static.posix/include" -o conftest conftest.cpp || exit 123
echo "------ done compiling ----"

./conftest || exit 123
rm -f ./conftest
rm -f ./conftest.cpp
echo "------ done running ----"
