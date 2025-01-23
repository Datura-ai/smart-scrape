import base64
import sys
import json
import bittensor as bt


def synapse_to_headers(self) -> dict:
    """
    Rewrite of the to_headers method to fix performance issues.
    Running get_required_fields everytime in loop caused significant delay.
    """

    # Initializing headers with 'name' and 'timeout'
    headers = {"name": self.name, "timeout": str(self.timeout)}

    # Adding headers for 'axon' and 'dendrite' if they are not None
    if self.axon:
        headers.update(
            {
                f"bt_header_axon_{k}": str(v)
                for k, v in self.axon.model_dump().items()
                if v is not None
            }
        )
    if self.dendrite:
        headers.update(
            {
                f"bt_header_dendrite_{k}": str(v)
                for k, v in self.dendrite.model_dump().items()
                if v is not None
            }
        )

    # Getting the fields of the instance
    instance_fields = self.model_dump()
    required = self.get_required_fields()

    # Iterating over the fields of the instance
    for field, value in instance_fields.items():
        # If the object is not optional, serializing it, encoding it, and adding it to the headers

        # Skipping the field if it's already in the headers or its value is None
        if field in headers or value is None:
            continue

        elif required and field in required:
            try:
                # create an empty (dummy) instance of type(value) to pass pydantic validation on the axon side
                serialized_value = json.dumps(value.__class__.__call__())
                encoded_value = base64.b64encode(serialized_value.encode()).decode(
                    "utf-8"
                )
                headers[f"bt_header_input_obj_{field}"] = encoded_value
            except TypeError as e:
                raise ValueError(
                    f"Error serializing {field} with value {value}. Objects must be json serializable."
                ) from e

    # Adding the size of the headers and the total size to the headers
    headers["header_size"] = str(sys.getsizeof(headers))
    headers["total_size"] = str(self.get_total_size())
    headers["computed_body_hash"] = self.body_hash

    return headers


class Synapse(bt.Synapse):
    def to_headers(self) -> dict:
        return synapse_to_headers(self)


class StreamingSynapse(bt.StreamingSynapse):
    def to_headers(self) -> dict:
        return synapse_to_headers(self)
