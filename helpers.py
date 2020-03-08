import numpy as np
class ExperienceBuffer():
    def __init__(self,size):
        self.size = size
        self.buffer = []
        self.pos    = -1
    def __len__(self):
        if self.pos == -1:
            return len(self.buffer)
        else:
            return self.size
    def append(self,entry):
        if self.pos == -1:
            self.buffer.append(entry)
            if len(self.buffer) == self.size:
                self.pos = 0
        else:
            self.buffer[self.pos] = entry
            self.pos = (self.pos+1)%self.size
    def sample(self,nEntry):
        indices = np.random.choice(len(self.buffer),nEntry,replace=False)
        return map(np.array,zip(*(self.buffer[idx] for idx in indices)))

