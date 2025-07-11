# BLAS Routines Reference

This docs provides a concise reference for ``pychop`` usage for the Basic Linear Algebra Subprograms (BLAS), categorized into three levels: Level 1 (vector-vector operations), Level 2 (matrix-vector operations), and Level 3 (matrix-matrix operations). These routines are essential for numerical linear algebra and are implemented in libraries like Intel MKL, OpenBLAS, and ATLAS. The following guide will lead users to call BLAS from  ``pychop``. 

**Note**: Routine names are listed without precision prefixes (S: single precision real, D: double precision real, C: single precision complex, Z: double precision complex). In practice, prefixes are required (e.g., `SAXPY`, `DGEMM`) to specify the data type.  BLAS routines require a prefix (S, D, C, Z) to indicate the data type. For example, `GEMM` becomes `SGEMM` for single precision real or `DGEMM` for double precision real.
- **Standards**: The tables cover standard BLAS routines as defined in the original specifications (Level 1: 1979, Level 2: 1984–1986, Level 3: 1987–1988). Non-standard extensions (e.g., sparse BLAS) are excluded.
- **Usage**: These routines are optimized for performance in numerical computing and are widely used in scientific computing, machine learning, and engineering applications.


## Level 1 BLAS: Vector-Vector Operations

| Routine   | Operation                     | Description                                           |
|-----------|-------------------------------|-------------------------------------------------------|
| AXPY      | $y = \alpha x + y$           | Adds a scaled vector to another vector.               |
| SCAL      | $x = \alpha x$               | Scales a vector by a scalar.                          |
| COPY      | $y = x$                      | Copies one vector to another.                         |
| SWAP      | $x \leftrightarrow y$        | Swaps two vectors.                                    |
| DOT       | $x \cdot y$                  | Computes the dot product of two vectors.              |
| DOTC/DOTU | $x \cdot \overline{y}$ or $x \cdot y$ | Complex conjugate/unconjugate dot product (complex). |
| NRM2      | $||x||_2$                    | Computes the Euclidean norm.                          |
| ASUM      | $||x||_1$                    | Computes the sum of absolute values (L1 norm).        |
| ROT       | Apply rotation               | Applies a Givens plane rotation to two vectors.       |
| ROTG      | Generate rotation            | Generates a Givens plane rotation.                    |
| ROTM      | Apply mod. rotation          | Applies a modified plane rotation.                    |
| ROTMG     | Generate mod. rotation       | Generates a modified plane rotation.                  |
| IAMAX     | Index of max                 | Finds the index of the maximum absolute value.        |
| IAMIN     | Index of min                 | Finds the index of the minimum absolute value.        |

## Level 2 BLAS: Matrix-Vector Operations

| Routine | Operation                              | Description                                                  |
|---------|----------------------------------------|--------------------------------------------------------------|
| GEMV    | $y = \alpha A x + \beta y$            | General matrix-vector multiplication (or with $A^T$).        |
| GBMV    | $y = \alpha A x + \beta y$            | General band matrix-vector multiplication.                   |
| SYMV    | $y = \alpha A x + \beta y$            | Symmetric matrix-vector multiplication.                      |
| SBMV    | $y = \alpha A x + \beta y$            | Symmetric band matrix-vector multiplication.                 |
| HEMV    | $y = \alpha A x + \beta y$            | Hermitian matrix-vector multiplication (complex).            |
| HBMV    | $y = \alpha A x + \beta y$            | Hermitian band matrix-vector multiplication.                 |
| SPMV    | $y = \alpha A x + \beta y$            | Symmetric matrix-vector multiplication (packed storage).     |
| HPMV    | $y = \alpha A x + \beta y$            | Hermitian matrix-vector multiplication (packed storage).     |
| TRMV    | $x = A x$                             | Triangular matrix-vector multiplication (or with $A^T$).    |
| TRSV    | $A x = b$                             | Solves triangular system (or with $A^T$).                   |
| TBMV    | $x = A x$                             | Triangular band matrix-vector multiplication.                |
| TBSV    | $A x = b$                             | Solves triangular band system.                              |
| TPMV    | $x = A x$                             | Triangular matrix-vector multiplication (packed storage).   |
| TPSV    | $A x = b$                             | Solves triangular system (packed storage).                  |
| GER     | $A = A + \alpha x y^T$                | Rank-1 update of a general matrix.                          |
| SYR     | $A = A + \alpha x x^T$                | Rank-1 update of a symmetric matrix.                        |
| SPR     | $A = A + \alpha x x^T$                | Rank-1 update of a symmetric matrix (packed storage).       |
| SYR2    | $A = A + \alpha x y^T + \alpha y x^T$ | Rank-2 update of a symmetric matrix.                        |
| SPR2    | $A = A + \alpha x y^T + \alpha y x^T$ | Rank-2 update of a symmetric matrix (packed storage).       |
| HER     | $A = A + \alpha x x^H$                | Rank-1 update of a Hermitian matrix (complex).              |
| HPR     | $A = A + \alpha x x^H$                | Rank-1 update of a Hermitian matrix (packed storage).       |
| HER2    | $A = A + \alpha x y^H + \alpha y x^H$ | Rank-2 update of a Hermitian matrix.                        |
| HPR2    | $A = A + \alpha x y^H + \alpha y x^H$ | Rank-2 update of a Hermitian matrix (packed storage).       |

## Level 3 BLAS: Matrix-Matrix Operations

| Routine | Operation                                     | Description                                                         |
|---------|-----------------------------------------------|---------------------------------------------------------------------|
| GEMM    | $C = \alpha \text{op}(A) \text{op}(B) + \beta C$ | General matrix-matrix multiplication, where $\text{op}(X) = X, X^T, X^H$. |
| SYMM    | $C = \alpha A B + \beta C$                   | Symmetric matrix-matrix multiplication (or $C = \alpha B A + \beta C$). |
| HEMM    | $C = \alpha A B + \beta C$                   | Hermitian matrix-matrix multiplication (or $C = \alpha B A + \beta C$). |
| SYRK    | $C = \alpha A A^T + \beta C$                 | Symmetric rank-k update (or $C = \alpha A^T A + \beta C$).         |
| HERK    | $C = \alpha A A^H + \beta C$                 | Hermitian rank-k update (or $C = \alpha A^H A + \beta C$).         |
| SYR2K   | $C = \alpha A B^T + \alpha B A^T + \beta C$  | Symmetric rank-2k update.                                          |
| HER2K   | $C = \alpha A B^H + \alpha B A^H + \beta C$  | Hermitian rank-2k update.                                          |
| TRMM    | $B = \alpha \text{op}(A) B$                  | Triangular matrix-matrix multiplication (or $B = \alpha B \text{op}(A)$). |
| TRSM    | $\text{op}(A) X = \alpha B$                  | Solves triangular system with multiple right-hand sides (or $X \text{op}(A) = \alpha B$). |

