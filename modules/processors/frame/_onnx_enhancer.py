*** Begin Patch
*** Update File: modules/processors/frame/_onnx_enhancer.py
@@
-import modules.globals
-from modules.platform_info import OPENVINO_PROVIDER_CONFIG
+import modules.globals
+from modules.platform_info import OPENVINO_PROVIDER_CONFIG
+import modules.runtime_profile as runtime_profile
@@
-    if providers is None:
-        providers = modules.globals.execution_providers
-
-    config = []
-    for p in providers:
-        if isinstance(p, tuple):
-            # Already configured – pass through
-            config.append(p)
-        elif p == "CUDAExecutionProvider":
-            # Use bare provider — ONNX Runtime's defaults are fastest on
-            # modern GPUs (Blackwell/sm_120).  Custom options like
-            # EXHAUSTIVE cudnn_conv_algo_search hurt performance on these
-            # architectures.
-            config.append(p)
-        elif p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
-            config.append((
-                "CoreMLExecutionProvider",
-                {
-                    "ModelFormat": "MLProgram",
-                    "MLComputeUnits": "ALL",
-                    "AllowLowPrecisionAccumulationOnGPU": 1,
-                },
-            ))
-        elif p == "OpenVINOExecutionProvider":
-            # AUTO lets OpenVINO select the best device
-            config.append(OPENVINO_PROVIDER_CONFIG)
-        else:
-            config.append(p)
-    return config
+    if providers is None:
+        providers = modules.globals.execution_providers
+
+    profile = runtime_profile.get_runtime_profile()
+    config = []
+    seen = set()
+    for p in providers:
+        if isinstance(p, tuple):
+            name = p[0]
+            if name not in seen:
+                config.append(p)
+                seen.add(name)
+            continue
+
+        name = p
+        if name == "CUDAExecutionProvider":
+            if profile["low_vram"]:
+                # Pin the device but avoid aggressive CUDA-specific tweaks
+                config.append(("CUDAExecutionProvider", {"device_id": 0}))
+            else:
+                config.append(name)
+        elif name == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
+            config.append((
+                "CoreMLExecutionProvider",
+                {
+                    "ModelFormat": "MLProgram",
+                    "MLComputeUnits": "ALL",
+                    "AllowLowPrecisionAccumulationOnGPU": 1,
+                },
+            ))
+        elif name == "OpenVINOExecutionProvider":
+            config.append(OPENVINO_PROVIDER_CONFIG)
+        else:
+            config.append(name)
+        seen.add(name)
+
+    # Ensure CPU fallback exists in low-vram mode to avoid crashes
+    if profile["low_vram"] and "CPUExecutionProvider" not in [
+        p[0] if isinstance(p, tuple) else p for p in config
+    ]:
+        config.append("CPUExecutionProvider")
+
+    return config
*** End Patch
