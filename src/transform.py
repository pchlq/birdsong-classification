import random

class FrequencyMask(object):
    def __init__(self, max_width, use_mean=bool(random.randint(0, 1))):
        self.max_width = max_width
        self.use_mean = use_mean
        
    def __call__(self, tensor_obj):
        tensor = tensor_obj.detach().clone()
        start = random.randrange(0, tensor.shape[2])
        end = start + random.randrange(1, self.max_width)
       
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor #[C, H, W]
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        format_string += "use_mean=" + (str(self.use_mean) + ")")
        return format_string


class TimeMask(object):
    def __init__(self, max_width, use_mean=bool(random.randint(0, 1))):
        self.max_width = max_width
        self.use_mean = use_mean
        
    def __call__(self, tensor_obj):
        tensor = tensor_obj.detach().clone()
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(1, self.max_width)
        
        if self.use_mean:
            tensor[:, :, start:end] = tensor.mean()
        else:
            tensor[:, :, start:end] = 0
        return tensor
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        format_string += " use_mean=" + (str(self.use_mean) + ")")
        return format_string


# if __name__ == "__main__":
#     print(FrequencyMask(max_width=34))