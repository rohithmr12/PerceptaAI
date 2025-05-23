import torch
import cv2
import numpy as np
from typing import List

def preprocess(img_rgb):
    """
    Preprocess RGB image for BSRGAN model inference.
    
    Args:
        img_rgb (numpy.ndarray): RGB image in HWC format (uint8)
    
    Returns:
        torch.Tensor: Preprocessed tensor in NCHW format
    """
    # Convert to float32 and normalize to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Convert HWC to CHW
    img_chw = np.transpose(img_float, (2, 0, 1))
    
    # Add batch dimension (NCHW)
    img_nchw = np.expand_dims(img_chw, axis=0)
    
    # Convert to torch tensor
    tensor = torch.from_numpy(img_nchw).float()
    
    return tensor

def postprocess(output_tensor):
    """
    Postprocess model output tensor to RGB image.
    
    Args:
        output_tensor: Model output tensor or numpy array
    
    Returns:
        numpy.ndarray: RGB image in HWC format (uint8)
    """
    # Convert to numpy if it's a tensor
    if isinstance(output_tensor, torch.Tensor):
        output_np = output_tensor.detach().cpu().numpy()
    else:
        output_np = output_tensor
    
    # Remove batch dimension if present (NCHW -> CHW)
    if output_np.ndim == 4:
        output_np = output_np.squeeze(0)
    
    # Convert CHW to HWC
    if output_np.ndim == 3:
        output_np = np.transpose(output_np, (1, 2, 0))
    
    # Normalize to [0, 255] and convert to uint8
    if output_np.dtype in [np.float32, np.float64]:
        # Clip values to valid range
        output_np = np.clip(output_np, 0.0, 1.0)
        output_np = (output_np * 255.0).astype(np.uint8)
    else:
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)
    
    return output_np

def collect_all_frames(video_capture) -> List[np.ndarray]:
    """
    Collect all frames from a video capture object.
    
    Args:
        video_capture (cv2.VideoCapture): OpenCV video capture object
    
    Returns:
        List[numpy.ndarray]: List of RGB frames (HWC format)
    """
    frames = []
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        print(f"Collected {len(frames)} frames from video")
        return frames
        
    except Exception as e:
        print(f"Error collecting frames: {e}")
        return frames

def write_all_frames(frames_rgb: List[np.ndarray], video_writer):
    """
    Write RGB frames to a video writer.
    
    Args:
        frames_rgb (List[numpy.ndarray]): List of RGB frames (HWC format)
        video_writer (cv2.VideoWriter): OpenCV video writer object
    """
    try:
        for i, frame_rgb in enumerate(frames_rgb):
            # Convert RGB to BGR for VideoWriter
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        print(f"Wrote {len(frames_rgb)} frames to video")
        
    except Exception as e:
        print(f"Error writing frames: {e}")

def pad_to_multiple(img, multiple=4):
    """
    Pad image to make dimensions divisible by multiple.
    
    Args:
        img (numpy.ndarray): Input image
        multiple (int): Multiple value for padding
    
    Returns:
        tuple: (padded_image, (pad_h, pad_w)) where pad_h and pad_w are padding amounts
    """
    h, w = img.shape[:2]
    
    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    # Apply padding
    if len(img.shape) == 3:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    
    return padded_img, (pad_h, pad_w)

def remove_padding(img, pad_info):
    """
    Remove padding from image.
    
    Args:
        img (numpy.ndarray): Padded image
        pad_info (tuple): (pad_h, pad_w) padding amounts
    
    Returns:
        numpy.ndarray: Image with padding removed
    """
    pad_h, pad_w = pad_info
    
    if pad_h == 0 and pad_w == 0:
        return img
    
    h, w = img.shape[:2]
    
    # Remove padding
    if pad_h > 0:
        img = img[:-pad_h, :]
    if pad_w > 0:
        img = img[:, :-pad_w]
    
    return img

def tile_process(img, tile_size=512, overlap=32):
    """
    Process large image by tiling (for memory efficiency).
    
    Args:
        img (numpy.ndarray): Input image
        tile_size (int): Size of each tile
        overlap (int): Overlap between tiles
    
    Returns:
        List[tuple]: List of (tile, position) tuples
    """
    h, w = img.shape[:2]
    tiles = []
    
    step = tile_size - overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract tile
            if len(img.shape) == 3:
                tile = img[y:y_end, x:x_end, :]
            else:
                tile = img[y:y_end, x:x_end]
            
            tiles.append((tile, (y, x, y_end, x_end)))
    
    return tiles

def merge_tiles(tiles_with_positions, output_shape, overlap=32, scale_factor=4):
    """
    Merge processed tiles back into a single image.
    
    Args:
        tiles_with_positions (List[tuple]): List of (processed_tile, position) tuples
        output_shape (tuple): Shape of the output image
        overlap (int): Overlap between tiles
        scale_factor (int): Scale factor applied to tiles
    
    Returns:
        numpy.ndarray: Merged image
    """
    if len(output_shape) == 3:
        h, w, c = output_shape
        merged = np.zeros((h * scale_factor, w * scale_factor, c), dtype=np.float32)
        weight_map = np.zeros((h * scale_factor, w * scale_factor, c), dtype=np.float32)
    else:
        h, w = output_shape
        merged = np.zeros((h * scale_factor, w * scale_factor), dtype=np.float32)
        weight_map = np.zeros((h * scale_factor, w * scale_factor), dtype=np.float32)
    
    for tile, (y, x, y_end, x_end) in tiles_with_positions:
        # Scale positions
        y_out = y * scale_factor
        x_out = x * scale_factor
        y_end_out = y_out + tile.shape[0]
        x_end_out = x_out + tile.shape[1]
        
        # Add tile to merged image
        merged[y_out:y_end_out, x_out:x_end_out] += tile.astype(np.float32)
        
        # Update weight map
        if len(output_shape) == 3:
            weight_map[y_out:y_end_out, x_out:x_end_out] += 1.0
        else:
            weight_map[y_out:y_end_out, x_out:x_end_out] += 1.0
    
    # Normalize by weight map
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    merged = merged / weight_map
    
    return merged.astype(np.uint8) 