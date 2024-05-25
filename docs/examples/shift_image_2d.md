```python
import torch
from torch_fourier_shift import fourier_shift_image_2d

# create a dummy image
my_image = torch.tensor(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]
)

# shift the image by 1 pixel in dim 0, 2 pixels in dim 1
shifted_image = fourier_shift_image_2d(image=my_image, shifts=torch.tensor([1, 2]))
```

## Notes
- shifts can be applied to arrays of 2D images `(..., h, w)`
- arrays of 2D shifts are supported `(..., 2)`

::: torch_fourier_shift.fourier_shift_image_2d

