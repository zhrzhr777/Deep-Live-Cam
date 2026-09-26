*** Begin Patch
*** Update File: modules/processors/frame/face_swapper.py
@@
-from modules.platform_info import OPENVINO_PROVIDER_CONFIG
+from modules.platform_info import OPENVINO_PROVIDER_CONFIG
+import modules.runtime_profile as runtime_profile
@@
 def get_face_swapper() -> Any:
     global FACE_SWAPPER
 
     with THREAD_LOCK:
         if FACE_SWAPPER is None:
@@
-            # Prefer FP16 on GPUs with Tensor Cores (Turing+) — half the
-            # memory bandwidth, faster inference.  Fall back to FP32 for
-            # older GPUs (e.g. GTX 16xx) where FP16 can produce NaN.
-            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
-            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
-            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)
-            if use_fp16:
-                model_path = fp16_path
-            elif os.path.exists(fp32_path):
-                model_path = fp32_path
-            else:
-                update_status(f"No inswapper model found in {models_dir}.", NAME)
-                return None
+            # Prefer FP16 / quantized variants when runtime requests them
+            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
+            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
+            quant_path = os.path.join(models_dir, "inswapper_128_quant.onnx")
+            profile = runtime_profile.get_runtime_profile()
+            model_path = None
+
+            # Order of preference: quantized -> fp16 -> fp32
+            if profile["prefer_fp16"] and os.path.exists(quant_path):
+                model_path = quant_path
+            elif profile["prefer_fp16"] and os.path.exists(fp16_path):
+                model_path = fp16_path
+            elif os.path.exists(fp32_path):
+                model_path = fp32_path
+            else:
+                update_status(f"No inswapper model found in {models_dir}.", NAME)
+                return None
@@
-                FACE_SWAPPER = insightface.model_zoo.get_model(
-                    model_path,
-                    providers=providers_config,
-                )
+                # Build session options via runtime_profile for ORT
+                # insightface.model_zoo.get_model accepts providers; ensure providers_config
+                # includes runtime-friendly options handled by _onnx_enhancer.build_provider_config
+                FACE_SWAPPER = insightface.model_zoo.get_model(
+                    model_path,
+                    providers=providers_config,
+                )
*** End Patch
