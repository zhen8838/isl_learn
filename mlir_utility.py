from typing import Callable
from mlir.ir import Module, AffineMap, Context, Operation, Block, Region, Value, OpView


class IrVisitor(object):
  def __init__(self, beforeBlock: Callable[[Block], bool] = None, beforeOperation: Callable[[OpView], bool] = None, afterBlock: Callable[[Block], bool] = None, afterOperation: Callable[[OpView], bool] = None) -> None:
    self.beforeBlockCallable = beforeBlock
    self.beforeOperationCallable = beforeOperation
    self.afterBlockCallable = afterBlock
    self.afterOperationCallable = afterOperation

  def visit(self, any):
    if isinstance(any, Block):
      return self.visitBlock(any)
    elif isinstance(any, Operation):
      return self.visitOperation(any.opview)
    elif isinstance(any, OpView):
      return self.visitOperation(any)
    elif isinstance(any, Module):
      return self.visitOperation(any.operation.opview)
    else:
      raise NotImplementedError()

  def visitBlock(self, block: Block):
    if not self.runBeforeBlock(block):
      return False
    for op in block.operations:
      if not self.visitOperation(op):
        return False
    if not self.runAfterBlock(block):
      return False
    return True

  def visitOperation(self, op: OpView):
    if not self.runBeforeOperation(op):
      return False
    for region in op.regions:
      for block in region.blocks:
        if not self.visitBlock(block):
          return False
    if not self.runAfterOperation(op):
      return False
    return True

  def runBeforeBlock(self, block: Block) -> bool:
    if self.beforeBlockCallable is not None:
      return self.beforeBlockCallable(block)
    return True

  def runBeforeOperation(self, op: OpView) -> bool:
    if self.beforeOperationCallable is not None:
      return self.beforeOperationCallable(op)
    return True

  def runAfterBlock(self, block: Block) -> bool:
    if self.afterBlockCallable is not None:
      return self.afterBlockCallable(block)
    return True

  def runAfterOperation(self, op: OpView) -> bool:
    if self.afterOperationCallable is not None:
      return self.afterOperationCallable(op)
    return True
