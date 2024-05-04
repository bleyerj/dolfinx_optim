import pytest
from ufl import as_matrix, as_vector, Identity, shape
from dolfinx_optim.utils import (
    to_vect,
    to_mat,
    hstack,
    vstack,
    get_slice,
    block_matrix,
    concatenate,
    tail,
)


@pytest.mark.parametrize("gdim", range(1, 4))
@pytest.mark.parametrize("tdim", range(1, 4))
def test_to_vect_to_mat(gdim, tdim):
    """Test to_vect and to_mat."""
    g = as_matrix([[tdim * i + j for j in range(tdim)] for i in range(gdim)])

    assert len(to_vect(g)) == tdim * gdim

    g2 = to_mat(to_vect(g), shape=(gdim, tdim))

    assert [g2[i, j] for i in range(gdim) for j in range(tdim)] == [
        g[i, j] for i in range(gdim) for j in range(tdim)
    ]


def test_slice():
    """Test UFL vector slicing."""
    X = as_vector([1, 2, 3, 4])
    assert tail(X) == as_vector([2, 3, 4])
    assert get_slice(X, end=2) == as_vector([1, 2])
    assert get_slice(X, end=-2) == as_vector([1, 2])
    assert get_slice(X, step=2) == as_vector([1, 3])


def test_stacking():
    """Test vertical and horizontal stacking."""
    Id = Identity(2)
    A = as_matrix([[1, 2], [3, 4]])
    B = as_matrix([[1, 2, 3], [4, 5, 6]])
    IAAI = as_matrix([[1, 0, 1, 2], [0, 1, 3, 4], [1, 2, 1, 0], [3, 4, 0, 1]])
    assert hstack([as_vector([0, 0]), A]) == as_matrix([[0, 1, 2], [0, 3, 4]])
    assert vstack([as_vector([0, 0]), A]) == as_matrix([[0, 0], [1, 2], [3, 4]])
    assert hstack([Id, A]) == as_matrix([[1, 0, 1, 2], [0, 1, 3, 4]])
    assert vstack([Id, A]) == as_matrix([[1, 0], [0, 1], [1, 2], [3, 4]])
    assert block_matrix([[Id, A], [A, Id]]) == IAAI
    assert shape(hstack([Id, B])) == (2, 5)
    assert shape(block_matrix([[Id, B], [B.T, Identity(3)]])) == (5, 5)


def test_concatenate():
    """Test vector concatenation."""
    x = [1, 2]
    assert concatenate(x) == as_vector(x)
    assert concatenate(x + [as_vector(x)]) == as_vector(x + x)
