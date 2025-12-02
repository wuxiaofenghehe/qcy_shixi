#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长图批量切片处理工具 - 支持多宽度自适应"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

try:
    from core.image_slicer import ImageSlicer
except ImportError:
    print("错误：无法导入ImageSlicer模块。请确保core/image_slicer.py文件存在。")
    sys.exit(1)


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
            print(f"  ⚠ 图像宽度 {width}px 使用最接近的配置 ({closest_width}px)")
        
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
    
    def get_image_files(self, input_dir: str) -> List[Path]:
        """获取目录中的所有图像文件"""
        input_path = Path(input_dir)
        return sorted([
            f for f in input_path.iterdir() 
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ])
    
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
            "image_width": width,
            "slice_config": {
                "block_size": config['SLICE_BLOCK_SIZE'],
                "overlap": config['SLICE_OVERLAP']
            },
            "saved_paths": saved_paths
        })
        
        slicer.clear_cache()
        return metadata
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     save_metadata: bool = True) -> None:
        """
        批量处理图像
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_metadata: 是否保存元数据
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = self.get_image_files(input_dir)
        if not image_files:
            print(f"在目录 {input_dir} 中未找到任何图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件\n")
        
        # 统计不同宽度的图像
        width_stats = {}
        
        # 处理每个图像
        metadata_list = []
        success_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                print(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")
                metadata = self.process_single_image(image_path, output_path)
                
                # 统计宽度
                width = metadata['image_width']
                width_stats[width] = width_stats.get(width, 0) + 1
                
                if save_metadata:
                    metadata_list.append(metadata)
                
                print(f"✓ 成功生成 {len(metadata['saved_paths'])} 个切片\n")
                success_count += 1
                
            except Exception as e:
                print(f"✗ 处理失败: {e}\n")
        
        # 保存元数据
        if save_metadata and metadata_list:
            self._save_metadata(metadata_list, output_path)
        
        # 显示统计信息
        print("=" * 50)
        print(f"处理完成！成功 {success_count}/{len(image_files)} 个图像")
        print("\n图像宽度统计:")
        for width, count in sorted(width_stats.items()):
            print(f"  {width}px: {count} 个")
        print("=" * 50)
    
    def _save_metadata(self, metadata_list: List[Dict], output_dir: Path) -> None:
        """保存元数据到JSON文件"""
        metadata_path = output_dir / "slicing_metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, indent=2, ensure_ascii=False)
            print(f"元数据已保存: {metadata_path}\n")
        except Exception as e:
            print(f"保存元数据失败: {e}\n")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='长图批量切片处理工具 - 自动适配360px和160px宽度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    args = parser.parse_args()
    
    # 配置输入输出路径
    input_dir = r"E:\qcy\new-data\新样本20251125"
    output_dir = r"E:\qcy\new-data\新样本20251125\sliced_images"
    
    # 验证输入目录
    if not os.path.isdir(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 构建自定义配置（仅包含用户指定的参数）
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
    
    # 显示配置信息
    print("=" * 50)
    print("自适应切片配置:")
    print("  支持宽度: 360px (块=360px, 重叠=30px)")
    print("            160px (块=160px, 重叠=30px)")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    if custom_config:
        print("\n  自定义覆盖:")
        for key, value in custom_config.items():
            print(f"    {key}: {value}")
    print("=" * 50 + "\n")
    
    # 执行批量处理
    processor = BatchImageSlicer(custom_config)
    processor.process_batch(
        input_dir, 
        output_dir, 
        save_metadata=not args.no_metadata
    )


if __name__ == "__main__":
    main()
