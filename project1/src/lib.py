import jaxlib.xla_extension as xla_ext
import jax._src.interpreters.ad as ad


def jax_type_to_python_type(value) -> float:
    if isinstance(value, float):
        pass
    elif isinstance(value, xla_ext.ArrayImpl):
        # print("value is xla_ext")
        value = value.item()
    elif isinstance(value, ad.JVPTracer):
        # print("value is JVPTracer")
        value = value.aval.val
    else:
        # print("value is type: ", type(value))
        raise Exception("Type not recognized: ", type(value))

    return value
