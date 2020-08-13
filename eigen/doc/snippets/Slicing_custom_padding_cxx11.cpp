struct pad {
  Index size() const { return out_size; }
  Index operator[] (Index i) const { return std::max<Index>(0,i-(out_size-in_size)); }
  Index in_size, out_size;
};

Matrix3i A;
A.reshaped() = VectorXi::LinSpaced(9,1,9);
cout << "Initial matrix A:\n" << A << "\n\n";
MatrixXi B(5,5);
B = A(pad{3,5}, pad{3,5});
cout << "A(pad{3,N}, pad{3,N}):\n" << B << "\n\n";
