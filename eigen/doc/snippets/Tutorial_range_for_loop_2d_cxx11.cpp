Matrix2i A = Matrix2i::Random();
cout << "Here are the coeffs of the 2x2 matrix A:\n";
for(auto x : A.reshaped())
  cout << x << " ";
cout << "\n";
