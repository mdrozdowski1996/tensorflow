/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/tensorrt/convert/utils.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {
namespace tensorrt {

bool IsGoogleTensorRTEnabled() {
  // TODO(laigd): consider also checking if tensorrt shared libraries are
  // accessible. We can then direct users to this function to make sure they can
  // safely write code that uses tensorrt conditionally. E.g. if it does not
  // check for for tensorrt, and user mistakenly uses tensorrt, they will just
  // crash and burn.
#if GOOGLE_CUDA && GOOGLE_TENSORRT
  return true;
#else
  return false;
#endif
}

Status GetPrecisionModeName(const int precision_mode, string* name) {
  switch (precision_mode) {
    case FP32MODE:
      *name = "FP32";
      break;
    case FP16MODE:
      *name = "FP16";
      break;
    case INT8MODE:
      *name = "INT8";
      break;
    default:
      return tensorflow::errors::OutOfRange("Unknown precision mode");
  }
  return Status::OK();
}

Status GetPrecisionMode(const string& name, int* precision_mode) {
  if (name == "FP32") {
    *precision_mode = FP32MODE;
  } else if (name == "FP16") {
    *precision_mode = FP16MODE;
  } else if (name == "INT8") {
    *precision_mode = INT8MODE;
  } else {
    return tensorflow::errors::InvalidArgument("Invalid precision mode name: ",
                                               name);
  }
  return Status::OK();
}

Status SerializeShapesString(
    const std::vector<tensorflow::TensorShape>& shapes, string* out,
    int max_batch_size) {
  std::ostringstream ss;
  for (int i = 0; i < shapes.size(); i++) {
    if (i != 0) ss << "\n";
    for (int j = 0; j < shapes[i].dims(); j++) {
      if (j != 0) ss << ",";
      // Use max_batch_size for batch dim
      if (max_batch_size != -1 && j == 0) {
        ss << max_batch_size;
      } else {
        ss << shapes[i].dim_size(j);
      }
    }
  }
  *out = ss.str();
  return tensorflow::Status::OK();
}

Status SerializeShapesString(
    const std::vector<tensorflow::TensorShapeProto>& shapes, string* out,
    int max_batch_size) {
  std::vector<TensorShape> tensor_shapes;
  for (const tensorflow::TensorShapeProto& proto : shapes) {
    tensor_shapes.emplace_back(proto);
  }
  return SerializeShapesString(tensor_shapes, out, max_batch_size);
}

Status DeserializeShape(const string& shape, TensorShape* out) {
  std::vector<int> dims;
  std::stringstream ss(shape);
  string string_dim;
  while (std::getline(ss, string_dim, ',')) {
    dims.push_back(std::stoi(string_dim));
  }
  TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(dims, out));
  return tensorflow::Status::OK();
}

Status DeserializeShapesString(const string& shapes,
                               std::vector<TensorShape>* out) {
  out->clear();
  std::stringstream ss(shapes);
  std::string serialized_shape;
  while (std::getline(ss, serialized_shape, '\n')) {
    TensorShape shape;

    LOG(INFO) << serialized_shape;
    TF_RETURN_IF_ERROR(DeserializeShape(serialized_shape, &shape));
    out->push_back(shape);
  }
  return tensorflow::Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow
