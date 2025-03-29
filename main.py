import numpy as np
import cv2
import tensorrt as trt
import ctypes
import pycuda.driver as cuda
from pycuda.driver import Stream
import os
import time

# ** 强制设置 UTF-8 编码 **
import sys
import locale
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class YOLOv5PostProcessor:
    def __init__(self, engine_path, conf_thres=0.5, iou_thres=0.45):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 初始化开始")
        self.conf_threshold = conf_thres
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   置信度阈值 (conf_threshold) = {self.conf_threshold}")
        self.iou_threshold = iou_thres
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   IOU 阈值 (iou_threshold) = {self.iou_threshold}")
        self.input_shape = (640, 640)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入形状 (input_shape) = {self.input_shape}")
        self.dynamic_allocation = False
        self.tensor_info = []
        self.bindings = None
        self.bindings_mem = []
        self.inputs = []
        self.outputs = []
        self.binding_volumes = {}

        # **显式创建和设置 CUDA Context**
        try:
            self.cuda_device = cuda.Device(0)
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] CUDA 设备初始化失败: {e}")
            raise
        self.cuda_context = self.cuda_device.make_context()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   显式创建 CUDA 上下文")

        # **新增流管理属性**
        self.stream = cuda.Stream()

        # 初始化 TensorRT 引擎
        self.logger = trt.Logger(trt.Logger.WARNING)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   初始化 TensorRT 日志记录器 (trt.Logger)")
        with open(engine_path, "rb") as f:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   从文件加载 Engine: {engine_path}")
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   Engine 反序列化完成")

        # 获取输入输出绑定索引 (使用 TensorRT 10.x Tensor API)
        self.input_name = "images"
        self.output_names = ["output0"]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入 Tensor 名称: {self.input_name}, 输出 Tensor 名称: {self.output_names}")

        self.input_idx = -1
        self.output_indices = []

        if not hasattr(self, 'bindings'):
            self.bindings = [None] * self.engine.num_io_tensors

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   处理 Tensor: '{tensor_name}'，索引: {i}")
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                if tensor_name == self.input_name:
                    self.input_idx = i
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     找到输入 Tensor '{self.input_name}'，索引: {self.input_idx}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     Tensor 模式: INPUT")
                tensor_mode = trt.TensorIOMode.INPUT
            elif self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                if tensor_name in self.output_names:
                    self.output_indices.append(i)
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     找到输出 Tensor '{tensor_name}'，索引: {i}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     Tensor 模式: OUTPUT")
                tensor_mode = trt.TensorIOMode.OUTPUT
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING]   Tensor 模式: UNKNOWN (理论上不应出现)")
                tensor_mode = trt.TensorIOMode.INPUT

            shape = list(self.engine.get_tensor_shape(tensor_name))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     Tensor 形状: {shape}")

            if any(dim < 0 for dim in shape):
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     !! 检测到动态维度: {shape}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     采用运行时动态内存分配策略")
                size = 0
                self.dynamic_allocation = True
            else:
                size = trt.volume(tuple(shape))

            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     Tensor 数据类型: {dtype}")

            self.tensor_info.append({
                'index': i,
                'name': tensor_name,
                'shape': shape,
                'dtype': dtype,
                'size': size,
                'mode': tensor_mode,
                'host_mem': None,
                'device_mem': None,
                'actual_shape': shape
            })
            binding_shape = self.engine.get_tensor_shape(tensor_name)
            volume = abs(trt.volume(binding_shape))
            self.binding_volumes[i] = volume * np.dtype(dtype).itemsize

        if self.input_idx == -1:
            raise ValueError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 输入 Tensor '{self.input_name}' 未在 Engine 中找到.")
        if not self.output_indices:
            raise ValueError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 输出 Tensor '{self.output_names}' 未在 Engine 中找到.")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入 Tensor '{self.input_name}' 的索引: {self.input_idx}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输出 Tensor '{self.output_names}' 的索引: {self.output_indices}")

        if not self.dynamic_allocation:
            self._allocate_buffers()

        # 创建执行上下文
        self.context = self.engine.create_execution_context()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   执行上下文创建完成")

        # 动态形状兼容性处理 (增强)
        if self.is_input_tensor(self.engine.get_tensor_name(self.input_idx)):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   Engine 支持动态形状 (从输入 Tensor 推断)")
            profile_min, profile_opt, profile_max = self.engine.get_tensor_profile_shape(
                name=self.engine.get_tensor_name(self.input_idx),
                profile_index=0
            )
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入 Tensor Profile: min={profile_min}, opt={profile_opt}, max={profile_max}")
            self.context.set_input_shape(self.engine.get_tensor_name(self.input_idx), profile_opt)
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   Engine 支持隐式批次维度 (可能是静态形状)")

        # 添加引擎信息诊断
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Engine 支持隐式批次维度: {not self.is_input_tensor(self.engine.get_tensor_name(self.input_idx))}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入绑定形状: {self.context.get_tensor_shape(self.engine.get_tensor_name(self.input_idx))}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输出绑定数量: {len(self.output_indices)}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入内存占用: {sum(info['host_mem'].nbytes for info in self.tensor_info if info['mode'] == trt.TensorIOMode.INPUT)/1024**2:.2f}MB" if not self.dynamic_allocation and self.tensor_info else "动态分配")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输出内存占用: {sum(info['host_mem'].nbytes for info in self.tensor_info if info['mode'] == trt.TensorIOMode.OUTPUT)/1024**2:.2f}MB" if not self.dynamic_allocation and self.tensor_info else "动态分配")
        total_device_mem_bytes = sum(info['device_mem'].size for info in self.tensor_info if info['device_mem']) if not self.dynamic_allocation and self.tensor_info else 0
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   总设备内存占用: {total_device_mem_bytes/1024**2:.2f}MB" if not self.dynamic_allocation else "动态分配")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 初始化完成")

    def __del__(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 析构函数开始")
        # 确保先释放 TensorRT 资源
        if hasattr(self, 'context'):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   释放 TensorRT 执行上下文")
            del self.context
            self.context = None # 显式设置为 None
        if hasattr(self, 'engine'):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   释放 TensorRT Engine")
            del self.engine
            self.engine = None # 显式设置为 None

        # 释放 CUDA 资源
        for info in self.tensor_info:
            if info['device_mem']:
                try:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   释放设备内存: {info['name']}")
                    info['device_mem'].free()
                except cuda.LogicError as e:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] 设备内存释放失败: {str(e)}")
                finally:
                    info['device_mem'] = None

        if hasattr(self, 'stream'):
            try:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   同步 CUDA Stream")
                self.stream.synchronize()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   释放 CUDA Stream")
                del self.stream
                self.stream = None # 显式设置为 None
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR]   CUDA Stream 释放异常: {str(e)}")

        if hasattr(self, 'cuda_context') and self.cuda_context:
            try:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   同步 CUDA Context")
                self.cuda_context.synchronize()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   弹出 CUDA Context")
                self.cuda_context.pop()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   分离 CUDA Context")
                self.cuda_context.detach()
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   CUDA 上下文资源释放完成")
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR]   CUDA 上下文释放异常: {str(e)}")
            finally:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   重置 CUDA Context 引用")
                self.cuda_context = None

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 析构函数结束")


    def is_input_tensor(self, tensor_name):
        return tensor_name == self.input_name

    def __call__(self, orig_image):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 前向推理 __call__ 函数开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入图像形状: {orig_image.shape}")
        resized_img = self.preprocess(orig_image)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   预处理后图像形状: {resized_img.shape}")

        current_input_shape = resized_img.shape
        if self.dynamic_allocation:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   动态内存分配模式，重新分配 buffers")
            self._allocate_buffers(batch_size=current_input_shape[0])

        if self.is_input_tensor(self.engine.get_tensor_name(self.input_idx)):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   使用 set_input_shape() 设置动态维度")
            self.context.set_input_shape(
                self.engine.get_tensor_name(self.input_idx),
                current_input_shape
            )
            current_shape = self.context.get_tensor_shape(self.engine.get_tensor_name(self.input_idx))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   动态维度已设置为: {current_shape}")
            assert current_shape == current_input_shape, f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 形状不匹配: 上下文 {current_shape} vs 输入 {current_input_shape}"

        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Stream 上下文已创建 (重用类级别 stream)")
            self.bindings = [None] * self.engine.num_io_tensors
            for info in self.tensor_info:
                if info['mode'] == trt.TensorIOMode.INPUT:
                    if not self.context.set_tensor_address(info['name'], int(info['device_mem'])):
                        raise RuntimeError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 设置输入 Tensor 地址失败，Tensor 名称: '{info['name']}'")
                    if info['host_mem'] is None:
                        raise RuntimeError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] 输入 Tensor '{info['name']}' 的 Host 内存未分配!")
                    input_host_mem = info['host_mem']
                    input_device_mem = info['device_mem']
                    self.bindings[info['index']] = int(info['device_mem'])
                elif info['mode'] == trt.TensorIOMode.OUTPUT:
                    if not self.context.set_tensor_address(info['name'], int(info['device_mem'])):
                        raise RuntimeError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 设置输出 Tensor 地址失败, Tensor 名称: '{info['name']}'")
                    if info['host_mem'] is None:
                        raise RuntimeError(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] 输出 Tensor '{info['name']}' 的 Host 内存未分配!")
                    output_host_mem = info['host_mem']
                    output_device_mem = info['device_mem']
                    self.bindings[info['index']] = int(info['device_mem'])

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   复制输入数据到 Host 内存")
            np.copyto(input_host_mem, resized_img.ravel())
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   异步复制 Host 内存到 Device 内存")
            cuda.memcpy_htod_async(input_device_mem, input_host_mem, self.stream)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   异步执行推理")
            try:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   执行时绑定指针: {self.bindings}")
                assert all(ptr is not None for ptr in self.bindings), f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL] 存在未初始化的设备指针!"
                self.context.execute_async_v3(
                    stream_handle=self.stream.handle
                )
            except RuntimeError as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] 推理失败: {str(e)}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   当前输入形状: {self.context.get_tensor_shape('images')}")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   设备内存状态: {[info['device_mem'].size if info['device_mem'] else 0 for info in self.tensor_info]}")
                raise
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   异步复制 Device 内存到 Host 内存")
            cuda.memcpy_dtoh_async(output_host_mem, output_device_mem, self.stream)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   同步 Stream")
            self.stream.synchronize()
            cuda.Context.synchronize()
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] __call__ 函数执行异常: {str(e)}")
            raise

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 推理完成")

        predictions = np.reshape(output_host_mem, self.tensor_info[self.output_indices[0]]['shape'])[0]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输出预测结果形状: {predictions.shape}")
        # ** 添加调试信息：打印模型输出 predictions 的前 10 行 **
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   模型输出 (predictions) 前 10 行:\n{predictions[:10]}")

        valid_detections = self.filter_predictions(predictions)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   置信度过滤后有效检测框形状: {valid_detections.shape}")

        final_boxes = self.non_max_suppression(valid_detections)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   NMS 处理后最终检测框形状: {final_boxes.shape}")

        scaled_boxes = self.scale_coordinates(final_boxes, orig_image.shape)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   坐标缩放后检测框形状: {scaled_boxes.shape}")

        result_img = self.draw_detections(orig_image, scaled_boxes)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]   检测框已绘制到图像上")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 前向推理 __call__ 函数结束")
        return result_img

    def _allocate_buffers(self, batch_size=1):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] _allocate_buffers 开始, batch_size={batch_size}")

        self.stream.synchronize()

        old_device_mems = {}
        for info in self.tensor_info:
            if info['device_mem']:
                old_device_mems[info['name']] = info['device_mem']
                info['device_mem'] = None

        self.bindings = [None] * self.engine.num_io_tensors

        for info in self.tensor_info:
            tensor_name = info['name']
            tensor_mode = info['mode']
            tensor_shape = list(info['shape'])
            dtype = info['dtype']

            if -1 in tensor_shape:
                tensor_shape[tensor_shape.index(-1)] = batch_size
            tensor_shape = tuple(tensor_shape)
            info['actual_shape'] = tensor_shape

            element_size = np.dtype(dtype).itemsize
            buffer_size = element_size * int(np.prod(tensor_shape))

            try:
                device_mem = cuda.mem_alloc(buffer_size)
            except cuda.CudaError as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR]   设备内存分配失败 (张量: {tensor_name}, 需求: {buffer_size//1024} KB): {str(e)}")
                raise
            info['device_mem'] = device_mem
            self.bindings[info['index']] = device_mem

            if tensor_mode == trt.TensorIOMode.INPUT:
                host_mem = cuda.pagelocked_empty(int(np.prod(tensor_shape)), dtype)
                info['host_mem'] = host_mem
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入[{info['index']}]分配页锁定内存: {int(np.prod(tensor_shape))} 元素 ({dtype}), 形状: {tensor_shape}")
            else:
                host_mem = np.empty(int(np.prod(tensor_shape)), dtype)
                info['host_mem'] = host_mem
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输出[{info['index']}]分配普通内存: {int(np.prod(tensor_shape))} 元素 ({dtype}), 形状: {tensor_shape}")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   绑定 {info['index']} ({tensor_name}) 分配设备内存: {buffer_size//1024} KB (基于形状{tensor_shape})")

        for tensor_name, mem in old_device_mems.items():
            if mem:
                try:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   释放旧设备内存: {tensor_name}")
                    mem.free()
                except cuda.LogicError as e:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] 旧内存释放异常: {str(e)}")
        self.stream.synchronize()
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] _allocate_buffers 结束\n")

    def preprocess(self, image):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 图像预处理 preprocess 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入图像形状: {image.shape}")
        h, w = image.shape[:2]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   原始图像高度: {h}, 宽度: {w}")
        scale = min(self.input_shape[0] / h, self.input_shape[1] / w)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   缩放比例: {scale}")

        resized = cv2.resize(image, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Resize 后图像形状: {resized.shape}")
        padded = np.full((*self.input_shape, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Padding 后图像形状: {padded.shape}")

        processed_image = padded.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   最终预处理图像形状 (带批次维度): {processed_image.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 图像预处理 preprocess 结束")
        return processed_image

    def filter_predictions(self, pred):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 预测结果过滤 filter_predictions 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入预测结果形状: {pred.shape}")
        obj_conf = pred[:, 4:5]
        cls_conf = pred[:, 5:6]
        scores = obj_conf * cls_conf
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   置信度分数形状: {scores.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     原始置信度分数 (前 5 个): {scores.flatten()[:5]}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   置信度阈值: {self.conf_threshold}")

        mask = scores.flatten() > self.conf_threshold
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Mask 形状: {mask.shape}, True 数量: {np.sum(mask)}")
        valid_preds = pred[mask]
        valid_scores = scores[mask]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     阈值过滤后有效置信度分数 (前 5 个): {valid_scores.flatten()[:5]}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   阈值过滤后有效预测框形状: {valid_preds.shape}")
        if valid_preds.size == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] !! 警告: 置信度阈值过滤后没有有效预测框 !!")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 预测结果过滤 filter_predictions 结束 (提前退出)")
            return np.empty((0, pred.shape[1]))

        boxes = self.xywh2xyxy(valid_preds[:, :4])
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   xywh2xyxy 转换后框形状: {boxes.shape}")
        cls_ids = np.zeros((boxes.shape[0], 1), dtype=np.float32)
        # ** 修改：调整拼接顺序，先 boxes, 再 cls_ids, 最后 valid_scores **
        filtered_detections = np.concatenate([boxes, cls_ids, valid_scores], axis=1)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   最终过滤检测结果形状: {filtered_detections.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 预测结果过滤 filter_predictions 结束")
        return filtered_detections

    def xywh2xyxy(self, x):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 坐标格式转换 xywh2xyxy 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入 x 形状: {x.shape}")
        y = np.empty_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 坐标格式转换 xywh2xyxy 结束")
        return y

    def non_max_suppression(self, detections):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 非极大值抑制 non_max_suppression 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   NMS 输入检测框形状: {detections.shape}")
        # ** 添加调试信息：打印 NMS 输入检测框的前 5 个置信度 **
        if detections.size > 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     NMS 输入检测框前 5 个置信度: {detections[:5, 5]}") # 修改为查看第6列（置信度列）

        if detections.size == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] !! 警告: NMS 输入为空，没有检测框 !!")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 非极大值抑制 non_max_suppression 结束 (提前退出)")
            return np.empty((0, 6))

        cls_boxes = detections
        if cls_boxes.size == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] !! 警告: 当前类别没有检测框，跳过 NMS.")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 非极大值抑制 non_max_suppression 结束 (提前退出)")
            return np.empty((0, 6))

        keep_boxes = [] # 初始化 keep_boxes

        # ** 修改：排序的列索引也要对应修改后的列顺序，置信度现在是第 6 列 (索引 5) **
        order = cls_boxes[:, 5].argsort()[::-1]
        cls_boxes = cls_boxes[order]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   按置信度排序后类别检测框形状: {cls_boxes.shape}")

        x1 = cls_boxes[:, 0]
        y1 = cls_boxes[:, 1]
        x2 = cls_boxes[:, 2]
        y2 = cls_boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        while cls_boxes.size > 0:
            keep = [] # 初始化 keep 放在循环内
            keep.append(cls_boxes[0])
            if cls_boxes.size == 1:
                break

            xx1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
            yy1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
            xx2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
            yy2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            iou = inter / (areas[0] + areas[1:] - inter)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   IOU 值 (前 5 个): {iou[:5]}")

            mask = iou <= self.iou_threshold
            cls_boxes = cls_boxes[1:][mask]
            areas = areas[1:][mask]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   IOU 过滤后剩余类别检测框形状: {cls_boxes.shape}")

            keep_boxes.extend(keep) # keep_boxes extend 放在循环外

        final_nms_boxes = np.array(keep_boxes) if keep_boxes else np.empty((0, 6))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   NMS 处理后最终检测框形状: {final_nms_boxes.shape}")
        # ** 添加调试信息：打印 NMS 输出检测框的前 5 个置信度 **
        if final_nms_boxes.size > 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     NMS 输出检测框前 5 个置信度: {final_nms_boxes[:5, 5]}") # 修改为查看第6列（置信度列）
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 非极大值抑制 non_max_suppression 结束")
        return final_nms_boxes

    def scale_coordinates(self, boxes, orig_shape):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 坐标缩放 scale_coordinates 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   输入检测框形状: {boxes.shape}")
        if boxes.size == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] !! 警告: scale_coordinates 输入为空，没有检测框 !!")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 坐标缩放 scale_coordinates 结束 (提前退出)")
            return np.empty((0, 6))

        orig_h, orig_w = orig_shape[:2]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   原始图像高度: {orig_h}, 宽度: {orig_w}")
        scale = min(self.input_shape[0] / orig_h, self.input_shape[1] / orig_w)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   缩放比例: {scale}")
        pad_x = (self.input_shape[1] - orig_w * scale) / 2
        pad_y = (self.input_shape[0] - orig_h * scale) / 2
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   Padding X 轴: {pad_x}, Padding Y 轴: {pad_y}")

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        np.clip(boxes[:, [0, 2]], 0, orig_w, out=boxes[:, [0, 2]])
        np.clip(boxes[:, [1, 3]], 0, orig_h, out=boxes[:, [1, 3]])

        # ** 修改: 只将坐标列转换为 int32, 保留置信度列 **
        scaled_boxes_int = boxes.copy() # 先复制一份，避免修改原始 boxes
        scaled_boxes_int[:, :4] = scaled_boxes_int[:, :4].astype(np.int32) # 只转换前 4 列 (x1, y1, x2, y2)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   缩放后整数坐标检测框形状: {scaled_boxes_int.shape}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 坐标缩放 scale_coordinates 结束")
        return scaled_boxes_int

    def draw_detections(self, image, boxes):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 绘制检测框 draw_detections 开始")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]   绘制检测框的输入形状: {boxes.shape}")
        if boxes.size == 0:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] !! 警告: draw_detections 输入为空，没有检测框可绘制 !!")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 绘制检测框 draw_detections 结束 (提前退出)")
            return image

        color = (0, 255, 0)

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            # ** 修改：置信度现在是第 6 列 (索引 5) **
            conf = box[5]
            cls_id = box[4] # 类别ID 现在是第 5 列 (索引 4)  (虽然这里没有用到cls_id，但为了完整性也修改了)
            label = f"player: {conf:.2f}"

            # ** 添加调试信息：打印 conf 值 **
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]     绘制检测框，置信度 (conf): {conf}")

            # ** 显式将 x1, y1, x2, y2 转换为 Python 的 int 类型 **
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(image, label, (int(x1), int(y1) - 10), # y 坐标也需要转换
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 绘制检测框 draw_detections 结束")
        return image

# 使用示例
if __name__ == "__main__":
    import pycuda.driver as cuda
    cuda.init()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] CUDA Driver 版本: {cuda.get_driver_version()}")

    processor = YOLOv5PostProcessor("best_fp16_int8.engine", conf_thres=0.7, iou_thres=0.3)
    input_folder = "images"
    output_folder = "output_test_images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    total_inference_time = 0
    count = 0

    processor.cuda_context.push()
    try:
        for image_file in image_files:
            input_path = os.path.join(input_folder, image_file)
            orig_img = cv2.imread(input_path)
            if orig_img is None:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] 警告: 无法读取图像文件 {image_file}, 跳过.")
                continue

            start_time = time.time()
            result_img = processor(orig_img)
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
            count += 1

            output_path = os.path.join(output_folder, f"output_{image_file}")
            cv2.imwrite(output_path, result_img)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 已处理图像: {image_file}, 推理时间: {inference_time:.3f}秒,  输出保存至: {output_path}")

        if count > 0:
            average_inference_time = total_inference_time / count
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] \n处理 {count} 张图像完成.")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 平均推理时间: {average_inference_time:.3f} 秒/每张图像")
        else:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] 没有找到或处理任何图像.")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] 所有图像处理完毕.")

    finally:
        processor.cuda_context.pop()
        del processor