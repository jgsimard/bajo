"""Common path float projection into CPU packet and GPU queue layouts."""

from layout import TileTensor
from bajo.rt.path_values import PathFloatField, PathValues


def store_path_values[
    Values: PathValues,
    origin: MutOrigin,
    //,
    STORE_BOUNDS: Bool = True,
    STORE_PDF: Bool = True,
](target: TileTensor[.float32, _, origin], values: Values, item: Int,):
    """Write only stage-live fields; bounds may be GPU scratch aliases."""
    comptime assert target.flat_rank == 2
    comptime for field in range(PathFloatField.VALUE_PLANES):
        comptime if (
            STORE_BOUNDS
            or field not in [PathFloatField.T_MIN, PathFloatField.T_MAX]
        ) and (STORE_PDF or field != PathFloatField.BSDF_PDF):
            target.store[width=Values.WIDTH](
                (field, item), values.float_field[field]()
            )
