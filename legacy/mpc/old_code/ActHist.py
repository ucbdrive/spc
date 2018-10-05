class ActHist:
    def __init__(self, size = 3):
        self.actions = []
        self.size = size

    def append(self, action):
        if len(self.actions) < self.size:
            self.actions.append(action)
        else:
            self.actions = self.actions[1:] + [action]
    
    def get_action(self):
        if len(self.actions) == self.size and self.actions[0] == self.actions[1] == self.actions[2]:
            return 4
        else:
            return self.actions[-1]
            
    def clear(self):
        self.actions = []