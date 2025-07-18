def solve_toeplitz(c_or_cr, b, check_finite=True):
    r"""Solve a Toeplitz system using Levinson Recursion

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    .. warning::

        Beginning in SciPy 1.17, multidimensional input will be treated as a batch,
        not ``ravel``\ ed. To preserve the existing behavior, ``ravel`` arguments
        before passing them to `solve_toeplitz`.

    Parameters
    ----------
    c_or_cr : array_like or tuple of (array_like, array_like)
        The vector ``c``, or a tuple of arrays (``c``, ``r``). If not
        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
        of the Toeplitz matrix is ``[c[0], r[1:]]``.
    b : (M,) or (M, K) array_like
        Right-hand side in ``T x = b``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (result entirely NaNs) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : (M,) or (M, K) ndarray
        The solution to the system ``T x = b``. Shape of return matches shape
        of `b`.

    See Also
    --------
    toeplitz : Toeplitz matrix

    Notes
    -----
    The solution is computed using Levinson-Durbin recursion, which is faster
    than generic least-squares methods, but can be less numerically stable.

    Examples
    --------
    Solve the Toeplitz system T x = b, where::

            [ 1 -1 -2 -3]       [1]
        T = [ 3  1 -1 -2]   b = [2]
            [ 6  3  1 -1]       [2]
            [10  6  3  1]       [5]

    To specify the Toeplitz matrix, only the first column and the first
    row are needed.

    >>> import numpy as np
    >>> c = np.array([1, 3, 6, 10])    # First column of T
    >>> r = np.array([1, -1, -2, -3])  # First row of T
    >>> b = np.array([1, 2, 2, 5])

    >>> from scipy.linalg import solve_toeplitz, toeplitz
    >>> x = solve_toeplitz((c, r), b)
    >>> x
    array([ 1.66666667, -1.        , -2.66666667,  2.33333333])

    Check the result by creating the full Toeplitz matrix and
    multiplying it by `x`.  We should get `b`.

    >>> T = toeplitz(c, r)
    >>> T.dot(x)
    array([ 1.,  2.,  2.,  5.])

    """
    # If numerical stability of this algorithm is a problem, a future
    # developer might consider implementing other O(N^2) Toeplitz solvers,
    # such as GKO (https://www.jstor.org/stable/2153371) or Bareiss.

    r, c, b, dtype, b_shape = _validate_args_for_toeplitz_ops(
        c_or_cr, b, check_finite, keep_b_shape=True)

    # accommodate empty arrays
    if b.size == 0:
        return np.empty_like(b)

    # Form a 1-D array of values to be used in the matrix, containing a
    # reversed copy of r[1:], followed by c.
    vals = np.concatenate((r[-1:0:-1], c))
    if b is None:
        raise ValueError('illegal value, `b` is a required argument')

    if b.ndim == 1:
        x, _ = levinson(vals, np.ascontiguousarray(b))
    else:
        x = np.column_stack([levinson(vals, np.ascontiguousarray(b[:, i]))[0]
                             for i in range(b.shape[1])])
        x = x.reshape(*b_shape)

    return x