from openfl.transport.grpc.client import CollaboratorGRPCClient, _atomic_connection, TensorRequest


class CollaboratorGRPCClient(CollaboratorGRPCClient):

    @_atomic_connection
    def get_tensor(self, collaborator_name, tensor_name, round_number,
                              report, tags, require_lossless):
        """Get tensor from the aggregator."""
        self._set_header(collaborator_name)
        request = TensorRequest(
            header=self.header,
            tensor_name=tensor_name,
            round_number=round_number,
            report=report,
            tags=tags,
            require_lossless=require_lossless
        )
        response = self.stub.GetTensor(request)
        # also do other validation, like on the round_number
        self.validate_response(response, collaborator_name)

        return response.tensor