import inspect
import torch
import numpy as np


def debug(*args):
    """
    Universal debug function with automatic variable name detection.
    
    Usage:
        debug(var)           # prints "var: <value>"
        debug(x, y, z)       # prints "x: <value>\\ny: <value>\\nz: <value>"
        debug(model.weight)  # prints "model.weight: <shape/value>"
    """
    frame = inspect.currentframe().f_back
    code = frame.f_code
    
    # Get the calling line
    import dis
    import re
    
    # Read the source line
    try:
        with open(code.co_filename, 'r') as f:
            lines = f.readlines()
            line = lines[frame.f_lineno - 1].strip()
            
            # Extract variable names from debug(...) call
            match = re.search(r'debug\((.*)\)', line)
            if match:
                var_names = [v.strip() for v in match.group(1).split(',')]
            else:
                var_names = [f"arg{i}" for i in range(len(args))]
    except:
        var_names = [f"arg{i}" for i in range(len(args))]
    
    # Print each variable with its name
    for name, value in zip(var_names, args):
        if isinstance(value, torch.Tensor):
            print(f"{name}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            if value.numel() <= 10:
                print(f"  values={value}")
        elif isinstance(value, np.ndarray):
            print(f"{name}: shape={value.shape}, dtype={value.dtype}")
            if value.size <= 10:
                print(f"  values={value}")
        elif hasattr(value, '__len__') and not isinstance(value, str):
            print(f"{name}: type={type(value).__name__}, len={len(value)}")
        else:
            print(f"{name}: {value}")
    print()  # Extra newline for readability


# Monkey-patch common types to add .debug() method
def _debug_method(self):
    """Debug method that can be called on any object"""
    frame = inspect.currentframe().f_back
    
    # Try to find the variable name
    var_name = None
    for name, val in frame.f_locals.items():
        if val is self:
            var_name = name
            break
    
    if var_name is None:
        var_name = "variable"
    
    if isinstance(self, torch.Tensor):
        print(f"{var_name}: shape={self.shape}, dtype={self.dtype}, device={self.device}")
        if self.numel() <= 10:
            print(f"  values={self}")
    elif isinstance(self, np.ndarray):
        print(f"{var_name}: shape={self.shape}, dtype={self.dtype}")
        if self.size <= 10:
            print(f"  values={self}")
    elif hasattr(self, '__len__') and not isinstance(self, str):
        print(f"{var_name}: type={type(self).__name__}, len={len(self)}")
    else:
        print(f"{var_name}: {self}")
    print()
    return self  # Return self for chaining


# Add .debug() to common types
def dimension_scaler(multiple:int, aspect_ratio_target: float=1.37, base_size: int = 16):
    """Generate dimensions that are multiples of `multiple` and roughly match `aspect_ratio_target`"""
    base_exact = multiple * base_size
    aspect_factor = aspect_ratio_target * base_exact // base_size
    aspect_match = int(aspect_factor*base_size)
    return base_exact, aspect_match


# Short alias
d = debug

if __name__ == "__main__":
    x, y = dimension_scaler(15)
    print(x, y, "n_voxels", f"{x * y:.3e}")
    
    x, y = dimension_scaler(12)
    print(x, y, "n_voxels", f"{x * y:.3e}")
    
    x, y = dimension_scaler(20)
    print(x, y, "n_voxels", f"{x * y:.3e}")