"""Common path float projection into CPU packet and GPU queue layouts."""

from layout import CoordLike
from bajo.rt.path_values import PathFloatField, PathValues
from bajo.core.tensor_layout import FieldTile


@always_inline
def store_path_values[
    Values: PathValues,
    origin: Origin[mut=True],
    Extent: CoordLike,
    fields: Int,
    //,
    STORE_BOUNDS: Bool = True,
    STORE_PDF: Bool = True,
](
    target: FieldTile[.float32, fields, Extent, origin],
    values: Values,
    item: Int,
):
    """Write only stage-live fields; bounds may be GPU scratch aliases."""
    comptime for field in range(PathFloatField.VALUE_PLANES):
        comptime if (
            STORE_BOUNDS
            or field not in [PathFloatField.T_MIN, PathFloatField.T_MAX]
        ) and (STORE_PDF or field != PathFloatField.BSDF_PDF):
            target.store[width=Values.WIDTH](
                (field, item), values.float_field[field]()
            )
