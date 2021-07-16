#include "mlir/mlir2onnx/mlir2onnx.h"
#include "mlir/ir/mlir_ops.h"
#include "onnx/onnx_pb.h"
#include <memory>
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ConversionPass : public mlir::OperationPass<ConversionPass> {
 public:
  void runOnOperation() final {
    getOperation()->dump();
    importModule(llvm::cast<mlir::ModuleOp>(getOperation()), &model_proto_);
  }

  void importModule(mlir::ModuleOp op, onnx::ModelProto* onnx_model) {
    auto main_func = op.lookupSymbol<mlir::FuncOp>("main");
    importFunction(main_func, onnx_model->mutable_graph());
  }
  void importFunction(mlir::FuncOp op, onnx::GraphProto* onnx_graph) {
    graph_node_ = onnx_graph;
    for (auto it : op.getArguments()) {
      auto vi = onnx_graph->add_input();
      value_map_[it] = vi;
      vi->set_name(toString(it));
    }
    for (auto& _op : op.front()) {
      auto node = onnx_graph->add_node();
      importOperation(&_op, node);
    }
    auto return_v = return_op_->getOperand(0);
    auto output_info = onnx_graph->add_output();
    output_info->CopyFrom(*value_map_[return_v]);
  }

  void importOperation(mlir::Operation* op, onnx::NodeProto* onnx_node,
                       onnx::GraphProto* onnx_graph = nullptr) {
    if (llvm::isa<mlir::ReturnOp>(op)) {
      return_op_ = op;
      return;
    };
    std::string node_name;
    if (auto ss_attr =
            op->getAttr("name").dyn_cast_or_null<mlir::StringAttr>()) {
      node_name = toString(ss_attr.getValue());
    } else {
      node_name = toString(op->getName().getStringRef());
      static size_t index = 0;
      node_name += std::to_string(index++);
    }
    onnx_node->set_name(node_name);
    for (auto attr : op->getAttrs()) {
      // auto onnx_attr = onnx_node->add_attribute();
      // onnx_attr->set_name(toString(attr.first));
      // onnx_attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      // onnx_attr->set_s(toString(attr.second));
    }
    auto result = op->getResult(0);
    addValueInfo(result);
    if (result.getType().isa<mlir::TupleType>()) {
      addValueInfo(result);
      for (auto user : result.getUsers()) {
        auto value_info = addValueInfo(user->getResult(0));
      }
    } else {
      onnx_node->add_output(*(addValueInfo(result)->mutable_name()));
    }
    for (auto operand : op->getOperands()) {
      if (value_map_.find(operand) == value_map_.end()) {
        throw std::runtime_error("");
      }
      onnx_node->add_input(*(value_map_[operand]->mutable_name()));
    }
  }
  ::onnx::ValueInfoProto* addValueInfo(mlir::Value vv) {
    value_map_[vv] = graph_node_->add_value_info();
    value_map_[vv]->set_name(toString(vv));
    return value_map_[vv];
  }

  const onnx::ModelProto& get_model_proto() const { return model_proto_; }

 private:
  std::string toString(mlir::Value vv) {
    if (vv.isa<mlir::BlockArgument>()) {
      static size_t index = 0;
      return "argv" + std::to_string(index++);
    }
    std::string buf;
    llvm::raw_string_ostream os(buf);
    vv.print(os);
    os.flush();
    auto pos = buf.find('=');
    if (pos != std::string::npos) {
      return buf.substr(0, pos);
    } else {
      return buf;
    }
  }

  // stringRef are not safe at last position may be not zero
  std::string toString(llvm::StringRef ref) {
    return ref.str();
  }

  std::string toString(mlir::Attribute attr) {
    std::string buf;
    llvm::raw_string_ostream os(buf);
    os << attr;
    os.flush();
    return buf;
  }

  mlir::Operation* return_op_ = nullptr;

  llvm::DenseMap<mlir::Value, ::onnx::ValueInfoProto*> value_map_;
  onnx::ModelProto model_proto_;
  onnx::GraphProto* graph_node_ = nullptr;
};

std::vector<char> mlir2OnnxModule(mlir::ModuleOp module) {
  std::vector<char> buf;
  mlir::PassManager pass_mgr(module.getContext());
  pass_mgr.addPass(std::make_unique<ConversionPass>());
  if (failed(pass_mgr.run(module))) {
    return buf;
  }
  auto proto =
      dynamic_cast<ConversionPass*>(&*pass_mgr.begin())->get_model_proto();
  auto buf_size = proto.ByteSizeLong();
  buf.resize(buf_size);
  proto.SerializeToArray(buf.data(), buf_size);
  return buf;
}
}