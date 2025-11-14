# Code Concepts Deep Dive - Train Llama Script

## Mục lục
1. [Dataclass vs Class thường](#1-dataclass-vs-class-thường)
2. [Type Hints và Optional](#2-type-hints-và-optional)
3. [Class Inheritance và Composition](#3-class-inheritance-và-composition)
4. [Path Object](#4-path-object)
5. [Advanced Concepts](#5-advanced-concepts)

---

## 1. Dataclass vs Class thường

### Class thường (Traditional Class)
```python
class FineTuneConfig:
    def __init__(self, model_name, max_seq_length, dtype, load_in_4bit):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
```

### Dataclass (Modern Python)
```python
@dataclass
class FineTuneConfig:
    model_name: str = "unsloth/Llama-3.1-8B-Instruct"
    max_seq_length: int = 8192
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False
```

### Tại sao dùng `@dataclass`?

#### **Ưu điểm:**

1. **Tự động sinh `__init__`**: Không cần viết constructor thủ công
   ```python
   # Dataclass tự động tạo:
   def __init__(self, model_name="unsloth/Llama-3.1-8B-Instruct", 
                max_seq_length=8192, ...):
       self.model_name = model_name
       self.max_seq_length = max_seq_length
       # ...
   ```

2. **Tự động sinh `__repr__`**: In object dễ đọc
   ```python
   config = FineTuneConfig()
   print(config)
   # Output: FineTuneConfig(model_name='unsloth/Llama-3.1-8B-Instruct', 
   #                        max_seq_length=8192, ...)
   ```

3. **Tự động sinh `__eq__`**: So sánh objects
   ```python
   config1 = FineTuneConfig()
   config2 = FineTuneConfig()
   print(config1 == config2)  # True
   ```

4. **Type hints rõ ràng**: Dễ debug, IDE hỗ trợ autocomplete tốt hơn

5. **Default values**: Giá trị mặc định ngay trong định nghĩa class

#### **Khi nào dùng dataclass?**
- **Configuration classes**: Lưu trữ config, settings
- **Data containers**: Chứa data không có logic phức tạp

#### **Khi nào dùng class thường?**
- **Logic phức tạp**: Nhiều methods, business logic
- **Inheritance phức tạp**: Kế thừa nhiều tầng
- **Validation logic**: Cần kiểm soát chặt chẽ việc khởi tạo
---

## 2. Type Hints và Optional

### `Optional[torch.dtype]` là gì?

```python
from typing import Optional
import torch

dtype: Optional[torch.dtype] = None
```

#### **Phân tích:**

1. **`Optional[X]`** = `Union[X, None]`
   ```python
   # Hai cách viết tương đương:
   dtype: Optional[torch.dtype] = None
   dtype: Union[torch.dtype, None] = None
   ```

2. **Ý nghĩa**:
   - Biến này có thể là `torch.dtype` (vd: `torch.float16`, `torch.bfloat16`)
   - Hoặc có thể là `None` (không chỉ định)

3. **Tại sao cần `Optional`?**
   ```python
   # Trường hợp 1: User chỉ định dtype rõ ràng
   config = FineTuneConfig(dtype=torch.bfloat16)
   
   # Trường hợp 2: Auto-detect (None)
   config = FineTuneConfig(dtype=None)  # Unsloth sẽ tự chọn dtype phù hợp
   ```

#### **Các kiểu dtype trong PyTorch:**
```python
torch.float32  # chinh xac - toc do cham 
torch.float16  # giam chinh xac - tang toc do
torch.bfloat16 # can bang giua ca 2
torch.int8     
```

#### **Type Hints khác trong code:**
```python
# String type
model_name: str = "..."

# Integer type
max_seq_length: int = 8192

# Boolean type
load_in_4bit: bool = False

# Float type
lora_dropout: float = 0.0

# Dictionary type
def load_datasets(self) -> Dict[str, Dataset]:
    # Trả về dictionary với key là string, value là Dataset
    return {"train": dataset1, "val": dataset2}

# Callable type (function)
field(default_factory=lambda: f"vietnamese-legal-llama-...")
# default_factory nhận một callable (function) để tạo giá trị mặc định
```
---

## 3. Class Inheritance và Composition

### Constructor `__init__` 

```python
class LlamaFineTuner:
    def __init__(self, config: FineTuneConfig, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        # ...
```

#### **Phân tích:**

1. **`def __init__(self, config: FineTuneConfig, data_dir: str)`**
   - `self`: Tham chiếu đến instance của class
   - `config: FineTuneConfig`: Parameter với type hint 
   - `data_dir: str`: Parameter kiểu string

2. **Đây là COMPOSITION, không phải INHERITANCE**
   ```python
   # COMPOSITION (Has-a relationship)
   class LlamaFineTuner:
       def __init__(self, config: FineTuneConfig):
           self.config = config  # LlamaFineTuner "có" một FineTuneConfig
   
   # INHERITANCE (Is-a relationship) - VÍ DỤ
   class LlamaFineTuner(BaseTrainer):  # LlamaFineTuner "là" một BaseTrainer
       def __init__(self):
           super().__init__()  # Gọi constructor của BaseTrainer
   ```

### Tại sao dùng separate Config class?

#### **Lý do 1: Separation of Concerns**
```python
# ❌ BAD: Tất cả trong một class
class LlamaFineTuner:
    def __init__(self, model_name, max_seq_length, lora_r, lora_alpha, 
                 batch_size, learning_rate, ...):  # 30+ parameters!
        self.model_name = model_name
        # ... rất dài và khó đọc

# ✅ GOOD: Tách biệt config và logic
class FineTuneConfig:
    # Chỉ chứa configuration
    model_name: str = "..."
    lora_r: int = 128

class LlamaFineTuner:
    # Chỉ chứa training logic
    def __init__(self, config: FineTuneConfig):
        self.config = config
```

#### **Lý do 2: Reusability**
```python
# Tạo nhiều configs khác nhau
config_h200 = FineTuneConfig(lora_r=128, batch_size=32)
config_a100 = FineTuneConfig(lora_r=64, batch_size=16)
config_t4 = FineTuneConfig(lora_r=32, batch_size=8)

# Dùng lại trainer với configs khác nhau
trainer1 = LlamaFineTuner(config_h200, data_dir)
trainer2 = LlamaFineTuner(config_a100, data_dir)
```

#### **Lý do 4: Validation**
```python
@dataclass
class FineTuneConfig:
    lora_r: int = 128
    
    def __post_init__(self):
        # Validate sau khi init
        if self.lora_r < 8 or self.lora_r > 256:
            raise ValueError("lora_r must be between 8 and 256")
```

---

## 4. Path Object

### `Path` vs `str` - Tại sao dùng `Path`?

```python
from pathlib import Path

# Code trong script
self.data_dir = Path(data_dir)
self.output_dir = Path(config.output_dir)
```

### So sánh `str` vs `Path`

#### **String (cách cũ)**
```python
import os

# ❌ Dùng string - phức tạp, dễ lỗi
data_dir = "/home/user/data"
config_file = data_dir + "/" + "config.json"  # Ugly concatenation
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

# Tạo thư mục
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
```

#### **Path (cách hiện đại)**
```python
from pathlib import Path

# ✅ Dùng Path - clean, safe, intuitive
data_dir = Path("/home/user/data")
config_file = data_dir / "config.json"  # Elegant operator overloading
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)

# Tạo thư mục
data_dir.mkdir(parents=True, exist_ok=True)  # Một lệnh, rõ ràng!
```

---

## 5. Advanced Concepts

### 5.3. List Comprehension & Generator Expressions

```python
# Trong code
config_dict = {k: v for k, v in config.__dict__.items() 
               if not k.startswith('_')}
```

#### **Dictionary Comprehension:**
```python
# Lọc các attribute không bắt đầu bằng '_'
config.__dict__  # {'model_name': '...', 'lora_r': 128, '_private': ...}
config_dict = {k: v for k, v in config.__dict__.items() 
               if not k.startswith('_')}
# {'model_name': '...', 'lora_r': 128}  # '_private' bị loại
```

#### **So sánh với loop thường:**
```python
# ❌ Traditional way - verbose
config_dict = {}
for k, v in config.__dict__.items():
    if not k.startswith('_'):
        config_dict[k] = v

# ✅ Comprehension - concise
config_dict = {k: v for k, v in config.__dict__.items() 
               if not k.startswith('_')}
```
### 5.5. String Formatting với f-strings

```python
# Code trong script
logger.info(f"📊 GPU Memory: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB used")
```

#### **Format specifiers:**
```python
value = 123.456789

f"{value}"           # "123.456789"
f"{value:.2f}"       # "123.46" (2 decimal places)
f"{value:.1f}"       # "123.5" (1 decimal place)
f"{value:,.2f}"      # "123.46" (thousands separator)
f"{value:>10.2f}"    # "    123.46" (right-aligned, width 10)
f"{value:0>10.2f}"   # "0000123.46" (zero-padded)

count = 1000000
f"{count:,}"         # "1,000,000" (thousands separator)
```

### 5.6. Lambda Functions

```python
# Code trong script
default_factory=lambda: f"vietnamese-legal-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

#### **Lambda vs Function:**
```python
# ❌ Regular function - verbose
def create_run_name():
    return f"vietnamese-legal-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

field(default_factory=create_run_name)

# ✅ Lambda - concise
field(default_factory=lambda: f"vietnamese-legal-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
```

#### **Khi nào dùng lambda?**
- Function đơn giản, 1 dòng
- Chỉ dùng 1 lần
- Không cần tên function


### 5.8. Unpacking với `**` operator

```python
# Loading config from dict
config_dict = {"model_name": "llama", "lora_r": 128}
config = FineTuneConfig(**config_dict)

# Tương đương:
config = FineTuneConfig(model_name="llama", lora_r=128)
```
