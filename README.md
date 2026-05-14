# Boolean Solver for DNF Formulas

This project checks whether a Boolean formula in **DNF** can be represented as

\[
\phi \equiv \phi_1 \wedge \neg \phi_2
\]

where:

- \(\phi_1\) is a **Horn-DNF**
- \(\phi_2\) is an **all-positive DNF**

The implementation follows a resolution-based method using CNF conversion, resolution closure, polarity-based partition, and consensus closure.

---

## Features

- Accepts Boolean formulas in **DNF**
- Converts DNF to CNF
- Computes **resolution closure**
- Splits the closure into:
  - \(\phi_N\): all-negative clauses
  - \(\phi\_{OP}\): clauses with at least one positive literal
- Builds:
  - \(\psi_2 := \neg \phi_N\)
  - \(\psi*1 := \mathrm{DNF}(\phi*{OP})\)
- Computes **consensus closure**
- Checks whether the result is **Horn-equivalent**
- Includes **unit tests**

---

## Requirements

- Python 3
- SymPy

Install SymPy with:

```bash
pip install sympy
```

or

```bash
python -m pip install sympy
```

Check installation:

```bash
python -c "import sympy; print(sympy.__version__)"
```

---

## Files

- `Boolean_solver.py` — main solver
- `test_Boolean_solver.py` — unit tests
- `README.md` — documentation

---

## Input format

The solver expects a DNF formula as a Python-style list of terms.

- each inner list is one **term**
- positive integer `k` means \(x_k\)
- negative integer `-k` means \(\neg x_k\)

Example:

```python
[[1, 2], [-1, 3]]
```

meaning

\[
(x_1 \wedge x_2) \vee (\neg x_1 \wedge x_3)
\]

Another example:

```python
[[1, -2, -3], [4]]
```

meaning

\[
(x_1 \wedge \neg x_2 \wedge \neg x_3) \vee x_4
\]

---

## Usage

### Run with menu mode

```bash
python3 Boolean_solver.py
```

### Run built-in examples

```bash
python3 Boolean_solver.py --example 1
python3 Boolean_solver.py --example 2
```

### Run with custom DNF input

```bash
python3 Boolean_solver.py --dnf "[[1,-2,-3],[4]]"
```

### Run in interactive mode

```bash
python3 Boolean_solver.py --interactive
```

### Show full details

```bash
python3 Boolean_solver.py --example 1 --full
```

### Print JSON output

```bash
python3 Boolean_solver.py --example 1 --json
```

---

## Built-in examples

```python
[[1, -2, -3], [4]]
[[1, 2], [-1, 3], [4]]
```

- Example 1: DNF, not Horn-DNF
- Example 2: Horn-DNF

---

## Run tests

```bash
python -m unittest test_Boolean_solver.py
```

or

```bash
python3 -m unittest test_Boolean_solver.py
```

Expected output:

```text
Ran 16 tests in 0.021s

OK
```

---

## Example results

### Example 1

```bash
python3 Boolean_solver.py --example 1
```

Input:

```python
[[1, -2, -3], [4]]
```

Expected:

- `psi1 Horn-DNF?` → NO
- Result → NO

### Example 2

```bash
python3 Boolean_solver.py --example 2
```

Input:

```python
[[1, 2], [-1, 3], [4]]
```

Expected:

- `psi1 Horn-DNF?` → YES
- Result → YES

---

## Common issue

If you get:

```text
Each term must be a list of integers.
```

make sure your input is a **list of terms**.

Wrong:

```python
[-3, -4]
```

Correct:

```python
[[-3, -4]]
```

---

## Output meaning

- `psi1 Horn-DNF?`  
  checks whether \(\psi_1\) is already Horn-DNF

- `RESULT`  
  shows whether the solver found a valid representation of the required form

A formula may fail because an essential non-Horn term remains after the consensus step.

---

## Report connection

This code supports the implementation described in the report.
It implements:

- DNF to CNF conversion
- resolution closure
- polarity-based partition
- construction of \(\psi_1\) and \(\psi_2\)
- consensus closure
- Horn-equivalence testing
