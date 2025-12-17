#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长图批量切片处理工具 - 支持多宽度自适应"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image


class ImageSlicer:
    """图像切片处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化切片器
        Args:
            config: 配置字典
        """
        self.config = config
        self.slice_block_size = config.get('SLICE_BLOCK_SIZE', 360)
        self.slice_overlap = config.get('SLICE_OVERLAP', 30)
        self.slice_start_y = config.get('SLICE_START_Y', 0)
        self.max_height = config.get('MAX_LONG_IMAGE_HEIGHT', 500000)
        self.max_width = config.get('MAX_LONG_IMAGE_WIDTH', 2048)
        self.memory_opt = config.get('SLICE_MEMORY_OPTIMIZATION', True)
        
        self.current_image = None
        self.slice_infos = []
        self.metadata = {}
    
    def slice_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        切片图像
        Args:
            image_path: 图像文件路径
        Returns:
            切片信息列表
        """
        try:
            with Image.open(image_path) as img:
                # 验证图像尺寸
                if img.height > self.max_height:
                    raise ValueError(f"图像高度 {img.height} 超过限制 {self.max_height}")
                if img.width > self.max_width:
                    raise ValueError(f"图像宽度 {img.width} 超过限制 {self.max_width}")
                
                # 存储原始图像信息
                self.metadata = {
                    'original_width': img.width,
                    'original_height': img.height,
                    'mode': img.mode,
                    'format': img.format
                }
                
                # 如果不需要内存优化，直接加载整个图像
                if not self.memory_opt:
                    self.current_image = img.copy()
                
                # 计算切片
                self.slice_infos = self._calculate_slices(img.height)
                
                # 如果使用内存优化，保存图像路径用于后续按需加载
                if self.memory_opt:
                    self.image_path = image_path
                else:
                    self.image_path = None
                
                return self.slice_infos
                
        except Exception as e:
            raise ValueError(f"切片图像失败: {e}")
    
    def _calculate_slices(self, image_height: int) -> List[Dict[str, Any]]:
        """
        计算切片信息
        Args:
            image_height: 图像高度
        Returns:
            切片信息列表
        """
        slices = []
        current_y = self.slice_start_y
        slice_index = 0
        
        while current_y < image_height:
            # 计算当前切片的结束位置
            end_y = min(current_y + self.slice_block_size, image_height)
            slice_height = end_y - current_y
            
            slice_info = {
                'index': slice_index,
                'start_y': current_y,
                'end_y': end_y,
                'height': slice_height,
                'width': self.metadata['original_width']
            }
            slices.append(slice_info)
            
            slice_index += 1
            
            # 如果已经到达图像底部，退出
            if end_y >= image_height:
                break
            
            # 计算下一个切片的起始位置（考虑重叠）
            current_y = end_y - self.slice_overlap
        
        return slices
    
    def save_slices(self, slice_infos: List[Dict[str, Any]], 
                   output_dir: str, base_name: str) -> List[str]:
        """
        保存切片图像
        Args:
            slice_infos: 切片信息列表
            output_dir: 输出目录
            base_name: 基础文件名
        Returns:
            保存的文件路径列表
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for slice_info in slice_infos:
            try:
                # 生成切片图像
                if self.memory_opt and self.image_path:
                    # 按需加载图像片段
                    with Image.open(self.image_path) as img:
                        slice_img = img.crop((
                            0,
                            slice_info['start_y'],
                            slice_info['width'],
                            slice_info['end_y']
                        ))
                else:
                    # 从已加载的图像中裁剪
                    slice_img = self.current_image.crop((
                        0,
                        slice_info['start_y'],
                        slice_info['width'],
                        slice_info['end_y']
                    ))
                
                # 保存切片
                output_file = output_path / f"{base_name}_slice_{slice_info['index']:03d}.png"
                slice_img.save(output_file, format='PNG')
                saved_paths.append(str(output_file))
                
            except Exception as e:
                print(f"  警告: 保存切片 {slice_info['index']} 失败: {e}")
        
        return saved_paths
    
    def get_slice_metadata(self) -> Dict[str, Any]:
        """获取切片元数据"""
        return {
            'original_size': (self.metadata['original_width'], 
                            self.metadata['original_height']),
            'slice_count': len(self.slice_infos),
            'slice_block_size': self.slice_block_size,
            'slice_overlap': self.slice_overlap,
            'slices': self.slice_infos
        }
    
    def clear_cache(self):
        """清除缓存"""
        self.current_image = None
        self.slice_infos = []


class BatchImageSlicer:
    """批量图像切片处理器"""
    
    # 不同宽度的预设配置
    WIDTH_CONFIGS = {
        360: {
            "SLICE_BLOCK_SIZE": 360,
            "SLICE_OVERLAP": 30,
        },
        160: {
            "SLICE_BLOCK_SIZE": 160,
            "SLICE_OVERLAP": 30,
        }
    }
    
    # 基础配置（所有宽度共享）
    BASE_CONFIG = {
        "SLICE_START_Y": 0,
        "MAX_LONG_IMAGE_HEIGHT": 500000,
        "MAX_LONG_IMAGE_WIDTH": 2048,
        "SLICE_MEMORY_OPTIMIZATION": True,
        "SLICE_BATCH_PROCESSING": True
    }
    
    # 支持的图像格式
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, custom_config: Dict[str, Any] = None):
        """
        初始化处理器
        Args:
            custom_config: 自定义配置，会覆盖默认配置
        """
        self.custom_config = custom_config or {}
        self.slicers = {}  # 缓存不同配置的slicer实例
    
    def get_image_width(self, image_path: Path) -> int:
        """获取图像宽度"""
        try:
            with Image.open(image_path) as img:
                return img.width
        except Exception as e:
            raise ValueError(f"无法读取图像尺寸: {e}")
    
    def get_config_for_width(self, width: int) -> Dict[str, Any]:
        """
        根据图像宽度获取对应配置
        Args:
            width: 图像宽度
        Returns:
            配置字典
        """
        # 查找最接近的预设宽度
        if width in self.WIDTH_CONFIGS:
            width_config = self.WIDTH_CONFIGS[width]
        else:
            # 找最接近的宽度配置
            closest_width = min(self.WIDTH_CONFIGS.keys(), 
                              key=lambda w: abs(w - width))
            width_config = self.WIDTH_CONFIGS[closest_width]
            print(f"图像宽度 {width}px 使用最接近的配置 ({closest_width}px)")
        
        # 合并配置：基础配置 + 宽度配置 + 自定义配置
        config = {
            **self.BASE_CONFIG,
            **width_config,
            **self.custom_config
        }
        
        return config
    
    def get_slicer(self, width: int) -> ImageSlicer:
        """
        获取或创建指定宽度的slicer实例
        Args:
            width: 图像宽度
        Returns:
            ImageSlicer实例
        """
        if width not in self.slicers:
            config = self.get_config_for_width(width)
            self.slicers[width] = ImageSlicer(config)
        return self.slicers[width]
    
    def get_image_files(self, input_path: str) -> List[Path]:
        """
        获取图像文件列表
        Args:
            input_path: 输入路径（文件或目录）
        Returns:
            图像文件路径列表
        """
        path = Path(input_path)
        
        if path.is_file():
            # 单个文件
            if path.suffix.lower() in self.IMAGE_EXTENSIONS:
                return [path]
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
        
        elif path.is_dir():
            # 目录中的所有图像文件
            return sorted([
                f for f in path.iterdir() 
                if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
            ])
        
        else:
            raise ValueError(f"无效的路径: {input_path}")
    
    def process_single_image(self, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        处理单个图像
        Args:
            image_path: 图像路径
            output_dir: 输出目录
        Returns:
            元数据字典
        """
        # 获取图像宽度
        width = self.get_image_width(image_path)
        print(f"  图像宽度: {width}px")
        
        # 获取对应的slicer
        slicer = self.get_slicer(width)
        config = self.get_config_for_width(width)
        print(f"  切片参数: 块={config['SLICE_BLOCK_SIZE']}px, "
              f"重叠={config['SLICE_OVERLAP']}px")
        
        # 切片处理
        slice_infos = slicer.slice_image(str(image_path))
        if not slice_infos:
            raise ValueError(f"无法切片图像: {image_path}")
        
        # 创建输出目录并保存
        image_output_dir = output_dir / image_path.stem
        saved_paths = slicer.save_slices(
            slice_infos, str(image_output_dir), image_path.stem
        )
        
        # 获取元数据
        metadata = slicer.get_slice_metadata()
        metadata.update({
            "image_path": str(image_path),
            "image_name": image_path.name,
            "image_width": width,
            "slice_config": {
                "block_size": config['SLICE_BLOCK_SIZE'],
                "overlap": config['SLICE_OVERLAP']
            },
            "saved_paths": saved_paths
        })
        
        slicer.clear_cache()
        return metadata
    
    def process_batch(self, input_path: str, output_dir: str) -> None:
        """
        批量处理图像
        Args:
            input_path: 输入路径（文件或目录）
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        try:
            image_files = self.get_image_files(input_path)
        except ValueError as e:
            print(f"错误: {e}")
            return
        
        if not image_files:
            print(f"在路径 {input_path} 中未找到任何图像文件")
            return
        
        # 判断是单文件还是批量处理
        is_single_file = len(image_files) == 1 and Path(input_path).is_file()
        
        if is_single_file:
            print(f"处理单个图像: {image_files[0].name}\n")
        else:
            print(f"找到 {len(image_files)} 个图像文件\n")
        
        # 统计不同宽度的图像
        width_stats = {}
        
        # 处理每个图像
        metadata_list = []
        success_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                if not is_single_file:
                    print(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")
                else:
                    print(f"处理: {image_path.name}")
                    
                metadata = self.process_single_image(image_path, output_path)
                
                # 统计宽度
                width = metadata['image_width']
                width_stats[width] = width_stats.get(width, 0) + 1
                
                
                print(f"成功生成 {len(metadata['saved_paths'])} 个切片\n")
                success_count += 1
                
            except Exception as e:
                print(f"处理失败: {e}\n")
        
        # 显示统计信息
        print("=" * 50)
        print(f"处理完成！成功 {success_count}/{len(image_files)} 个图像")
        if width_stats:
            print("\n图像宽度统计:")
            for width, count in sorted(width_stats.items()):
                print(f"  {width}px: {count} 个")
        print("=" * 50)

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='长图批量切片处理工具 - 自动适配360px和160px宽度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('input', type=str, 
                       help='输入路径（图像文件或包含图像的目录）')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出目录路径')
    
    # 可选参数
    parser.add_argument('--block-size', type=int, help='切片块大小(像素)')
    parser.add_argument('--overlap', type=int, help='切片重叠区域大小(像素)')
    parser.add_argument('--start-y', type=int, help='切片起始Y坐标(像素)')
    parser.add_argument('--max-height', type=int, help='最大图像高度限制(像素)')
    parser.add_argument('--max-width', type=int, help='最大图像宽度限制(像素)')
    parser.add_argument('--no-memory-opt', action='store_true', help='禁用内存优化')
    parser.add_argument('--no-batch', action='store_true', help='禁用批处理模式')
    parser.add_argument('--no-metadata', action='store_true', help='不保存元数据文件')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    
    # 如果没有命令行参数，使用默认配置
    if len(sys.argv) == 1:
        # 默认配置
        input_path = r"E:\qcy\new-data\新样本20251125"
        output_dir = r"E:\qcy\new-data\新样本20251125\sliced_images"
        custom_config = {}
        save_metadata = True
        
        print("使用默认配置运行...")
    else:
        # 解析命令行参数
        args = parser.parse_args()
        input_path = args.input
        output_dir = args.output
        save_metadata = not args.no_metadata
        
        # 构建自定义配置
        custom_config = {}
        if args.block_size is not None:
            custom_config["SLICE_BLOCK_SIZE"] = args.block_size
        if args.overlap is not None:
            custom_config["SLICE_OVERLAP"] = args.overlap
        if args.start_y is not None:
            custom_config["SLICE_START_Y"] = args.start_y
        if args.max_height is not None:
            custom_config["MAX_LONG_IMAGE_HEIGHT"] = args.max_height
        if args.max_width is not None:
            custom_config["MAX_LONG_IMAGE_WIDTH"] = args.max_width
        if args.no_memory_opt:
            custom_config["SLICE_MEMORY_OPTIMIZATION"] = False
        if args.no_batch:
            custom_config["SLICE_BATCH_PROCESSING"] = False
    
    # 验证输入路径
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 显示配置信息
    print("自适应切片配置:")
    print("  支持宽度: 360px (块=360px, 重叠=30px)")
    print("            160px (块=160px, 重叠=30px)")
    
    # 判断输入类型
    if os.path.isfile(input_path):
        print(f"  输入文件: {input_path}")
    else:
        print(f"  输入目录: {input_path}")
    
    print(f"  输出目录: {output_dir}")
    
    if custom_config:
        print("\n  自定义覆盖:")
        for key, value in custom_config.items():
            print(f"    {key}: {value}")
    
    # 执行批量处理
    processor = BatchImageSlicer(custom_config)
    processor.process_batch(input_path, output_dir, save_metadata=save_metadata)


if __name__ == "__main__":
    main()
