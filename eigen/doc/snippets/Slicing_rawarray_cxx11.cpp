#if EIGEN_HAS_STATIC_ARRAY_TEMPLATE
MatrixXi A = MatrixXi::Random(4,6);
cout << "Initial matrix A:\n" << A << "\n\n";
cout << "A(all,{4,2,5,5,3}):\n" << A(all,{4,2,5,5,3}) << "\n\n";
#endif
