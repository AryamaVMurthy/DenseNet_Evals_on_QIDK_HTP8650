- Assumes tiny-imagenet-200 exists
- ONNX/DLC files need to be downloaded from qualcomm hub
- Assumes qaihub is setup


1. ONNX may be optimised by qaihub to a more compile-friendly version
2. ONNX needs to be quantised
3. Quantised/Unquantised ONNX models must be compiled to DLC's (or DLC's for some quantisations can
directly be downloaded from qaihub)
4. DLC's need to be linked to context binaries (single DLC -> context binary is the only relevant
workflow in this context)
5. context binaries may be deployed on the NPU