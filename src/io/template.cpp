#include <iostream>

//template <typename A>
class BB {

};

template <typename A, typename B>
class C {
 public:
    template <typename D>
    B a();
};

template <typename A, typename B>
template <typename D>
B C<A, B>::a() {}

int main() {
    C<int, BB> c = C<int, BB>();

    c.a();

    return 0;

}