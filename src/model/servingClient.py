import os
import sys
from ast import literal_eval

import tensorflow as tf
import numpy as np

from grpc.beta import implementations

# reference local copy of Tensorflow Serving API Files
sys.path.append(os.path.join(os.getcwd(), os.pardir, os.pardir, 'ext_libs'))
import lib.predict_pb2 as predict_pb2
import lib.prediction_service_pb2 as prediction_service_pb2


class ServingClient:

    def __init__(self, host, port, model_info, inputs_info, timeout=10.0):
        self.host = host
        self.port = int(port)
        self.channel = implementations.insecure_channel(host, self.port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

        self.model_info = model_info
        self.inputs_info = inputs_info
        self.timeout = timeout

    def predict(self, x):
        # TOFIX not generic
        self.inputs_info[0]['value'] = x
        res = self._predict(self.model_info, self.inputs_info)
        #print("Results " + str(res))
        #print("Results shape " + str(res.shape))
        return res

    def _predict(self, model_info, inputs_info):
        #print("Inputs info" + str(inputs_info))
        #print("Inputs info shape" + str(inputs_info[0]['value'].shape))
        request = self._build_request(model_info, inputs_info)
        # call prediction on the server
        #print("Sending request " + str(request))
        results = self.stub.Predict(request, timeout=self.timeout)

        return ServingClient._transform_results(results)

    def _build_request(self, model_info, inputs_info):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_info['name']
        request.model_spec.signature_name = model_info['signature_name']

        # define inputs
        for cur_input in inputs_info:
            cur_input_tensor = tf.contrib.util.make_tensor_proto(cur_input['value'],
                                                        dtype=tf.float32 if cur_input['type']=="float32" else tf.int64,
                                                        shape=cur_input['value'].shape)
            request.inputs[cur_input['name']].CopyFrom(cur_input_tensor)
        return request

    # TODO generalize for N outputs
    @staticmethod
    def _transform_results(results):
        # get the output from server response
        outputs = results.outputs['outputs']
        # extract response-tensor shape
        tensor_shape = outputs.tensor_shape
        tensor_shape = [dim.size for dim in tensor_shape.dim]
        # reshape list of float to given shape
        res_tensor = np.array(outputs.float_val).reshape(tensor_shape)

        return res_tensor
