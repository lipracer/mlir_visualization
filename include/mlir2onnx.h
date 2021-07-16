#ifndef INCLUDE_MLIRTOONNX_H
#define INCLUDE_MLIRTOONNX_H

#include <vector>
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Analysis/Dominance.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"


namespace mlir {
std::vector<char> mlir2OnnxModule(mlir::ModuleOp module);
}

#endif