import unittest

# Import the functions we want to test from the main solver file.
# Each imported function will be checked with small unit tests.
from Boolean_solver import (
    canon_lits,
    negate_lit,
    negate_clause,
    all_positive,
    all_negative,
    is_horn_clause,
    is_horn_term,
    highest_var,
    is_consistent_literals,
    remove_redundant_terms,
    remove_redundant_clauses,
    consensus_term,
    consensus_closure,
    equivalent_dnf,
    solve_formula,
)


# This class groups all unit tests for the Boolean solver.
# Each test function checks one small part of the program.
class TestBooleanSolver(unittest.TestCase):

    # Test negation of one literal.
    # Positive should become negative, and negative should become positive.
    def test_negate_lit(self):
        self.assertEqual(negate_lit(3), -3)
        self.assertEqual(negate_lit(-2), 2)

    # Test negation of all literals inside one clause.
    # This checks whether clause negation is working correctly.
    def test_negate_clause(self):
        self.assertEqual(negate_clause([3, 4]), [-3, -4])
        self.assertEqual(negate_clause([-1, 4, -5]), [1, -4, 5])

    # Test canonical ordering of literals.
    # This also checks that duplicate literals are removed.
    def test_canon_lits(self):
        self.assertEqual(canon_lits([3, -1, 3, 2]), (-1, 2, 3))

    # Test whether a list contains only positive literals.
    def test_all_positive(self):
        self.assertTrue(all_positive([1, 2, 4]))
        self.assertFalse(all_positive([1, -2, 4]))

    # Test whether a list contains only negative literals.
    def test_all_negative(self):
        self.assertTrue(all_negative([-1, -2, -4]))
        self.assertFalse(all_negative([1, -2, -4]))

    # Test Horn clause checking.
    # A Horn clause can have at most one positive literal.
    def test_is_horn_clause(self):
        self.assertTrue(is_horn_clause([-1, -2, 3]))
        self.assertTrue(is_horn_clause([-1, -2]))
        self.assertFalse(is_horn_clause([1, 2, -3]))

    # Test Horn term checking.
    # A Horn-DNF term can have at most one negative literal.
    def test_is_horn_term(self):
        self.assertTrue(is_horn_term([1, 2]))
        self.assertTrue(is_horn_term([-1, 2, 3]))
        self.assertFalse(is_horn_term([1, -2, -3]))

    # Test detection of the largest variable index in a formula.
    # This also checks empty input cases.
    def test_highest_var(self):
        self.assertEqual(highest_var([[1, -2], [4, 5]]), 5)
        self.assertEqual(highest_var([]), 0)
        self.assertEqual(highest_var([[]]), 0)

    # Test consistency of literals.
    # A list is inconsistent if it contains both x and -x.
    def test_is_consistent_literals(self):
        self.assertTrue(is_consistent_literals([1, 2, -3]))
        self.assertFalse(is_consistent_literals([1, -1, 3]))

    # Test removal of redundant DNF terms.
    # This checks duplicates, subsumed terms, and inconsistent terms.
    def test_remove_redundant_terms(self):
        dnf = [[1, 2], [1, 2], [1, 2, 3], [1, -1]]
        self.assertEqual(remove_redundant_terms(dnf), [[1, 2]])

    # Test removal of redundant CNF clauses.
    # This checks duplicates, subsumed clauses, and inconsistent clauses.
    def test_remove_redundant_clauses(self):
        cnf = [[1, 2], [1, 2], [1, 2, 3], [1, -1]]
        self.assertEqual(remove_redundant_clauses(cnf), [[1, 2]])

    # Test consensus of two DNF terms.
    # First pair should produce a valid consensus term.
    # Other pairs should return None because they do not satisfy the rule.
    def test_consensus_term(self):
        self.assertEqual(consensus_term([1, 2], [-1, 2]), [2])
        self.assertIsNone(consensus_term([1, 2], [-1, -2]))
        self.assertIsNone(consensus_term([1, 2], [3, 4]))

    # Test full consensus closure.
    # Starting from two terms, the closure should simplify to [[2]].
    def test_consensus_closure(self):
        dnf = [[1, 2], [-1, 2]]
        self.assertEqual(consensus_closure(dnf), [[2]])

    # Test logical equivalence between two DNFs.
    # The first pair is equivalent.
    # The second pair is not equivalent.
    def test_equivalent_dnf(self):
        self.assertTrue(equivalent_dnf([[1], [2]], [[2], [1]]))
        self.assertFalse(equivalent_dnf([[1]], [[2]]))

    # Test a known example that should fail for the Boolean solver.
    # It should not produce a successful final result.
    def test_solve_formula_fail_example(self):
        result = solve_formula([[1, -2, -3], [4]])
        self.assertFalse(result.success)
        self.assertFalse(result.psi1_is_horn)

    # Test a known example that should succeed for the Boolean solver.
    # It should produce success and valid phi1 and phi2.
    def test_solve_formula_success_example(self):
        result = solve_formula([[1, 2], [-1, 3], [4]])
        self.assertTrue(result.success)
        self.assertTrue(result.horn_equivalent_after_consensus)
        self.assertIsNotNone(result.phi1)
        self.assertIsNotNone(result.phi2)


# This runs all unit tests when the file is executed directly.
if __name__ == "__main__":
    unittest.main()


# To run the tests, execute this command in the terminal:
# python3 -m unittest test_Boolean_solver.py
#
# This will run all the test cases defined in the TestBooleanSolver class
# and report the results.
