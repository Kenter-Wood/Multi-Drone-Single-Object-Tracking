import struct
import zlib
from pathlib import Path
import binascii  # 添加用于十六进制显示

def check_png_signature(file_path):
    """检查PNG文件签名"""
    png_signature = b'\x89PNG\r\n\x1a\n'
    with open(file_path, 'rb') as f:
        header = f.read(8)
        return header == png_signature

def analyze_chunks(file_path):
    """解析并显示所有数据块信息"""
    print("\n正在解析PNG数据块结构...")
    with open(file_path, 'rb') as f:
        f.read(8)  # 跳过文件头
        
        chunk_id = 0
        while True:
            chunk_id += 1
            # 读取块长度 (4字节大端)
            length_bytes = f.read(4)
            if not length_bytes:
                break
                
            length = struct.unpack('>I', length_bytes)[0]
            chunk_type_bytes = f.read(4)  # 保存原始字节
            chunk_type = chunk_type_bytes.decode('ascii', errors='replace')  # 使用replace处理可能的解码错误
            
            print(f"\n块 #{chunk_id}: {chunk_type}")
            print(f"  长度: {length} 字节")
            print(f"  块类型原始字节: {binascii.hexlify(chunk_type_bytes).decode()}")
            
            # 显示位置信息
            position = f.tell()
            print(f"  数据开始位置: 字节 {position}")
            
            # 读取数据
            data = f.read(length)
            if len(data) < length:
                print(f"  警告: 数据不足! 预期 {length} 字节，实际读取 {len(data)} 字节")
            
            # 读取CRC
            crc_bytes = f.read(4)
            if len(crc_bytes) < 4:
                print(f"  错误: CRC数据不完整! 读取了 {len(crc_bytes)} 字节")
                break
                
            # 计算并验证CRC
            crc_calculated = zlib.crc32(chunk_type_bytes + data) & 0xFFFFFFFF  # 修复：使用字节类型
            crc_stored = struct.unpack('>I', crc_bytes)[0]
            
            print(f"  CRC 存储值: 0x{crc_stored:08x}")
            print(f"  CRC 计算值: 0x{crc_calculated:08x}")
            print(f"  CRC 校验: {'通过' if crc_calculated == crc_stored else '失败'}")
            
            # 显示数据头部(最多16字节)用于调试
            if length > 0:
                display_bytes = min(16, length)
                print(f"  数据预览 (前{display_bytes}字节): {binascii.hexlify(data[:display_bytes]).decode()}")
            
            # 特殊处理关键数据块
            if chunk_type == 'IHDR':
                if length >= 13:
                    width, height, bit_depth, color_type = struct.unpack('>IIBB', data[:10])
                    print(f"  图像尺寸: {width}x{height}")
                    print(f"  位深度: {bit_depth}")
                    print(f"  颜色类型: {color_type} ({get_color_type_desc(color_type)})")
                else:
                    print(f"  错误: IHDR块数据不足，无法解析图像信息")
            elif chunk_type == 'IDAT':
                print(f"  压缩数据长度: {len(data)} 字节")
            elif chunk_type == 'IEND':
                print("  文件结束标记")
                
            if chunk_type == 'IEND':
                break

def get_color_type_desc(color_type):
    """获取颜色类型描述"""
    types = {
        0: "灰度图像",
        2: "真彩色图像",
        3: "索引彩色图像",
        4: "带Alpha通道的灰度图像",
        6: "带Alpha通道的真彩色图像"
    }
    return types.get(color_type, "未知类型")

def extract_idat_data(file_path):
    """提取并合并所有IDAT数据块"""
    idat_data = bytearray()
    with open(file_path, 'rb') as f:
        f.read(8)  # 跳过文件头
        
        while True:
            length_bytes = f.read(4)
            if not length_bytes:
                break
                
            length = struct.unpack('>I', length_bytes)[0]
            chunk_type = f.read(4).decode('ascii', errors='replace')
            
            data = f.read(length)
            f.read(4)  # 跳过CRC
            
            if chunk_type == 'IDAT':
                idat_data.extend(data)
                
            if chunk_type == 'IEND':
                break
                
    return bytes(idat_data)

def main():
    file_path = "origin_0_20250420-150117_3.png"
    
    if not Path(file_path).exists():
        print("错误：文件不存在")
        return
    
    # 第一步：检查文件签名
    if not check_png_signature(file_path):
        print("\n文件签名错误！可能原因：")
        print("- 文件不是有效的PNG格式")
        print("- 文件头损坏或被修改")
        return
    
    print("\nPNG文件签名验证通过")
    
    # 第二步：解析数据块
    try:
        analyze_chunks(file_path)
        
        # 第三步：分析IDAT数据
        idat_data = extract_idat_data(file_path)
        print(f"\n提取的IDAT数据总大小: {len(idat_data)} 字节")
        
        # 尝试解压IDAT数据
        try:
            decompressed = zlib.decompress(idat_data)
            print(f"解压后的数据大小: {len(decompressed)} 字节 (解压成功)")
            print(f"前32字节预览: {binascii.hexlify(decompressed[:32]).decode()}")
        except zlib.error as e:
            print(f"IDAT数据解压失败: {str(e)}")
            print("这表明图像数据已损坏或不完整")
            
    except Exception as e:
        print(f"\n解析过程中发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("可能原因：")
        print("- 数据块长度不正确")
        print("- 文件提前结束")
        print("- CRC校验失败导致结构损坏")

if __name__ == "__main__":
    main()