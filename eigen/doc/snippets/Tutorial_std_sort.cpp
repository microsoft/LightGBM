Array4i v = Array4i::Random().abs();
cout << "Here is the initial vector v:\n" << v.transpose() << "\n";
std::sort(v.begin(), v.end());
cout << "Here is the sorted vector v:\n" << v.transpose() << "\n";
