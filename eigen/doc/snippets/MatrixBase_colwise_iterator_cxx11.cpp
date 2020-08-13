Matrix3i m = Matrix3i::Random();
cout << "Here is the initial matrix m:" << endl << m << endl;
int i = -1;
for(auto c: m.colwise()) {
  c *= i;
  ++i;
}
cout << "Here is the matrix m after the for-range-loop:" << endl << m << endl;
auto cols = m.colwise();
auto it = std::find_if(cols.cbegin(), cols.cend(),
                       [](Matrix3i::ConstColXpr x) { return x.squaredNorm() == 0; });
cout << "The first empty column is: " << distance(cols.cbegin(),it) << endl;
