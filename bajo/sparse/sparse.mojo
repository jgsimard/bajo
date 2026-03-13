from std.memory import memset_zero, memcpy
from std.testing import assert_true


struct SparseMatrix[
    type: DType,
    itype: DType where itype.is_integral(),
    is_triplet: Bool,
    has_values: Bool,
]:
    """A generic sparse matrix in Compressed Column or Triplet form."""

    var nzmax: Int
    """Maximum number of entries."""
    var m: Int
    """Number of rows."""
    var n: Int
    """Number of columns."""
    var p: UnsafePointer[Scalar[Self.itype], MutExternalOrigin]
    """Column pointers (size n+1) or col indices (size nzmax)."""
    var i: UnsafePointer[Scalar[Self.itype], MutExternalOrigin]
    """Row indices, size nzmax."""
    var x: UnsafePointer[Scalar[Self.type], MutExternalOrigin]
    """Numerical values, size nzmax."""
    var nz: Int
    """Number of entries if triplet, -1 if CCS."""

    def __init__(out self, m: Int, n: Int, nzmax: Int):
        debug_assert[assert_mode="all"](
            m >= 0 and n >= 0, "Matrix dimensions must be non-negative"
        )

        self.m = m
        self.n = n
        self.nzmax = max(nzmax, 1)

        comptime if Self.is_triplet:
            self.nz = 0
            self.p = alloc[Scalar[Self.itype]](self.nzmax)
        else:
            self.nz = -1
            var p = alloc[Scalar[Self.itype]](n + 1)
            p[0] = 0
            self.p = p

        self.i = alloc[Scalar[Self.itype]](self.nzmax)

        comptime if Self.has_values:
            self.x = alloc[Scalar[Self.type]](self.nzmax)
        else:
            self.x = {}

    def __del__(deinit self):
        self.p.free()
        self.i.free()
        self.x.free()

    def nnz(self) -> Int:
        comptime if self.is_triplet:
            return self.nz
        else:
            return Int(self.p[self.n])

    def _flip(self, i: Int) -> Int:
        return -i - 2

    def _unflip(self, i: Int) -> Int:
        return self._flip(i) if i < 0 else i

    def _marked(self, j: Int) -> Bool:
        return self.p[j] < 0

    def _mark(mut self, j: Int):
        self.p[j] = self._flip(Int(self.p[j]))

    def add_entry(
        mut self,
        i: Scalar[Self.itype],
        j: Scalar[Self.itype],
        value: Scalar[Self.type],
    ):
        comptime assert Self.is_triplet, (
            "add_entry can only be called on a SparseMatrix in Triplet form"
            " (is_triplet=True)"
        )

        self.i[self.nz] = i
        self.p[self.nz] = j
        self.x[self.nz] = value

        self.nz += 1
        self.m = max(self.m, Int(i + 1))
        self.n = max(self.n, Int(j + 1))


def main():
    print("hello")
