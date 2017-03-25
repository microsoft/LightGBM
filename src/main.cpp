#include <iostream>
#include <LightGBM/application.h>

int main(int argc, char** argv) {
  LightGBM::Application app(argc, argv);
  app.Run();
}
