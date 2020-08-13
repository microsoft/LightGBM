#ifndef TEST_SOLVERBASE_H
#define TEST_SOLVERBASE_H

template<typename DstType, typename RhsType, typename MatrixType, typename SolverType>
void check_solverbase(const MatrixType& matrix, const SolverType& solver, Index rows, Index cols, Index cols2)
{
  // solve
  DstType m2               = DstType::Random(cols,cols2);
  RhsType m3               = matrix*m2;
  DstType solver_solution  = DstType::Random(cols,cols2);
  solver._solve_impl(m3, solver_solution);
  VERIFY_IS_APPROX(m3, matrix*solver_solution);
  solver_solution          = DstType::Random(cols,cols2);
  solver_solution          = solver.solve(m3);
  VERIFY_IS_APPROX(m3, matrix*solver_solution);
  // test solve with transposed
  m3                       = RhsType::Random(rows,cols2);
  m2                       = matrix.transpose()*m3;
  RhsType solver_solution2 = RhsType::Random(rows,cols2);
  solver.template _solve_impl_transposed<false>(m2, solver_solution2);
  VERIFY_IS_APPROX(m2, matrix.transpose()*solver_solution2);
  solver_solution2         = RhsType::Random(rows,cols2);
  solver_solution2         = solver.transpose().solve(m2);
  VERIFY_IS_APPROX(m2, matrix.transpose()*solver_solution2);
  // test solve with conjugate transposed
  m3                       = RhsType::Random(rows,cols2);
  m2                       = matrix.adjoint()*m3;
  solver_solution2         = RhsType::Random(rows,cols2);
  solver.template _solve_impl_transposed<true>(m2, solver_solution2);
  VERIFY_IS_APPROX(m2, matrix.adjoint()*solver_solution2);
  solver_solution2         = RhsType::Random(rows,cols2);
  solver_solution2         = solver.adjoint().solve(m2);
  VERIFY_IS_APPROX(m2, matrix.adjoint()*solver_solution2);
}

#endif // TEST_SOLVERBASE_H
