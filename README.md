- [Netron](https://netron.app/)是一款优秀的绘图软件，支持onnx、hlo、pytouch、ncnn等等众多机器学习模型，目前版本没有支持mlir。
- mlir Module可以序列化以及反序列化，即op的Parse与Print但是mlir需要依赖具体Dialect的定义，因此猜想netron只支持mlir StandardOp，或是诸如tf mhlo这种 Dialect