std::vector<int> ind{4,2,5,5,3};
MatrixXi A = MatrixXi::Random(4,6);
cout << "Initial matrix A:\n" << A << "\n\n";
cout << "A(all,ind):\n" << A(all,ind) << "\n\n";
